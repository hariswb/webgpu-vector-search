import { ShardGPURecord, ShardTopK } from "./types";

export class GpuSimilarityEngine {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  // shard cache: shardIndex -> GPU resource record
  private shardCache: Map<number, ShardGPURecord> = new Map();

  // total GPU bytes currently allocated for shards
  private totalGpuBytes = 0;

  // user-configurable maximum GPU bytes to allocate for shards (default 512 MB)
  private maxGpuBytes = 256 * 1024 * 1024;

  // Reusable buffers (created after init)
  private paramsBuffer: GPUBuffer | null = null; // uniform (dim, count)
  private queryBuffer: GPUBuffer | null = null; // storage for query vector (maxDim sized)
  private readbackBuffer: GPUBuffer | null = null; // mapped READ buffer for scores
  private maxDim = 0;
  private maxShardCount = 0; // track largest shard count to size readback

  constructor(maxGpuBytes?: number) {
    if (maxGpuBytes) this.maxGpuBytes = maxGpuBytes;
  }

  async init(dimHint = 1024, maxShardCountHint = 8192): Promise<void> {
    if (!("gpu" in navigator)) {
      throw new Error("WebGPU not supported in this browser.");
    }
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) throw new Error("Failed to request GPU adapter.");

    this.device = await this.adapter.requestDevice();
    if (!this.device) throw new Error("need a browser that supports WebGPU");

    // Create shader module & pipeline (same shader as in your original file)
    const shaderModule = this.device.createShaderModule({
      label: "Compute Vector Similarity",
      code: /* wgsl */ `
        struct Params {
          dim: u32,
          count: u32
        };
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> query: array<f32>;
        @group(0) @binding(2) var<storage, read> vectors: array<f32>;
        @group(0) @binding(3) var<storage, read_write> out: array<f32>;

        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let i = gid.x;
          if (i >= params.count) {
            return;
          }

          var sum: f32 = 0.0;
          let d: u32 = params.dim;
          let base: u32 = i * d;
          var k: u32 = 0u;
          loop {
            if (k >= d) { break; }
            sum = sum + query[k] * vectors[base + k];
            k = k + 1u;
          }
          out[i] = sum;
        }
      `,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    // setup reusable buffers according to hints (will be resized on demand)
    this.maxDim = Math.max(1, dimHint);
    this.maxShardCount = Math.max(1, maxShardCountHint);

    // uniform buffer: two u32 (dim, count) -> 8 bytes. We'll allocate 256 bytes aligned.
    const uniformSize = 256; // safe aligned size
    this.paramsBuffer = this.device.createBuffer({
      size: uniformSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // query buffer: allocate maxDim * 4 bytes
    this.queryBuffer = this.device.createBuffer({
      size: this.maxDim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // create initial readback buffer sized for maxShardCount
    const readbackByteLength = this.maxShardCount * 4;
    this.readbackBuffer = this.device.createBuffer({
      size: readbackByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }


    // ---------------------------
  // Shard upload & residency
  // ---------------------------

  /**
   * Upload a shard to GPU and create bind group; uses mappedAtCreation to avoid staging copy.
   * - shardIndex: numeric ID of shard (should be unique, e.g., manifest index)
   * - shardVectors: Float32Array with length == dim * count
   */
  async uploadShardToGPU(
    shardIndex: number,
    shardVectors: Float32Array,
    dim: number,
    count: number,
    startIndex: number
  ): Promise<void> {
    if (!this.device || !this.bindGroupLayout) {
      throw new Error("Engine not initialized. Call init() first.");
    }

    const byteSize = shardVectors.byteLength;

    // If we already have this shard resident, update metadata and return.
    const existing = this.shardCache.get(shardIndex);
    if (existing) {
      // update metadata if necessary
      existing.count = count;
      existing.startIndex = startIndex;
      existing.lastUsed = Date.now();
      return;
    }

    // Evict if adding this shard would exceed budget
    await this.evictIfNeeded(byteSize);

    // Create GPU buffer with mappedAtCreation true for faster upload
    const gpuBuffer = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    // Write directly into the mapped range (zero-copy path)
    {
      const mappedRange = gpuBuffer.getMappedRange();
      const mappedF32 = new Float32Array(mappedRange as ArrayBuffer);
      mappedF32.set(shardVectors);
      gpuBuffer.unmap();
    }

    // Cache gpuBuffer for shard
    const record: ShardGPURecord = {
      shardIndex,
      count,
      startIndex,
      byteSize,
      gpuBuffer,
      // leave bindGroup null for now; we'll create full bind group at compute time (fast)
      bindGroup: null as unknown as GPUBindGroup,
      lastUsed: Date.now(),
    };

    this.shardCache.set(shardIndex, record);
    this.totalGpuBytes += byteSize;

    // update resizing of reuse buffers if this shard is bigger than previous max
    if (count > this.maxShardCount) {
      await this.resizeReadbackBuffer(count);
    }
  }

  /**
   * Evict least recently used shards until there is enough room for 'neededBytes'
   */
  private async evictIfNeeded(neededBytes: number) {
    if (!this.device) return;
    if (this.totalGpuBytes + neededBytes <= this.maxGpuBytes) return;

    // Build a list of records sorted by lastUsed ascending (oldest first)
    const entries = Array.from(this.shardCache.values()).sort((a, b) => a.lastUsed - b.lastUsed);

    for (const rec of entries) {
      // destroy GPU buffer where supported
      try {
        // Some browsers support GPUBuffer.destroy()
        (rec.gpuBuffer as any).destroy?.();
      } catch (e) {
        // ignore
      }
      this.shardCache.delete(rec.shardIndex);
      this.totalGpuBytes -= rec.byteSize;

      if (this.totalGpuBytes + neededBytes <= this.maxGpuBytes) return;
    }

    // If we evicted everything and still not enough, throw
    if (this.totalGpuBytes + neededBytes > this.maxGpuBytes) {
      throw new Error("Not enough GPU memory for shard upload after eviction.");
    }
  }

  // Resize the global readback buffer to accommodate the largest shard count
  private async resizeReadbackBuffer(newMaxCount: number) {
    if (!this.device) throw new Error("No device");

    this.maxShardCount = newMaxCount;
    // Destroy old readback buffer if any
    try {
      (this.readbackBuffer as any)?.destroy?.();
    } catch (e) {}

    const readbackByteLength = this.maxShardCount * 4;
    this.readbackBuffer = this.device.createBuffer({
      size: readbackByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  // ---------------------------
  // Compute using resident shard
  // ---------------------------

  /**
   * Ensure queryBuffer and paramsBuffer are large enough for given dim and writes params & query
   */
  private ensureQueryAndParams(dim: number, count: number, query: Float32Array) {
    if (!this.device || !this.queryBuffer || !this.paramsBuffer) throw new Error("Not initialized");

    // If query buffer not large enough, recreate it (rare)
    if (dim > this.maxDim) {
      try {
        (this.queryBuffer as any).destroy?.();
      } catch {}
      this.maxDim = dim;
      this.queryBuffer = this.device.createBuffer({
        size: this.maxDim * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }

    // Write query contents into queryBuffer (only first dim floats)
    this.device.queue.writeBuffer(this.queryBuffer, 0, query.buffer, query.byteOffset, dim * 4);

    // Write params (two u32: dim, count) into paramsBuffer (at offset 0)
    const paramsArray = new Uint32Array([dim >>> 0, count >>> 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);
  }

  /**
   * Compute dot products for a shard that is resident on GPU (no re-upload of vectors).
   * Returns Float32Array scores of length == count.
   */
  private async computeShardUsingResidentBuffer(
    shardRec: ShardGPURecord,
    dim: number,
  ): Promise<Float32Array> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout || !this.queryBuffer || !this.paramsBuffer || !this.readbackBuffer) {
      throw new Error("Engine not initialized.");
    }

    const count = shardRec.count;
    const outByteLength = count * 4;

    // create an output storage buffer for this dispatch (we can't bind the global readback buffer directly
    // as a storage buffer; we use a GPU-side storage buffer then copy into readbackBuffer)
    const outBuffer = this.device.createBuffer({
      size: outByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // create bind group for this dispatch: bind params (0), query (1), shard vectors (2), out (3).
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer! } },
        { binding: 1, resource: { buffer: this.queryBuffer! } },
        { binding: 2, resource: { buffer: shardRec.gpuBuffer } },
        { binding: 3, resource: { buffer: outBuffer } },
      ],
    });

    // encode commands
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);

    const workgroupSize = 64;
    const dispatchX = Math.ceil(count / workgroupSize);
    pass.dispatchWorkgroups(dispatchX);
    pass.end();

    // copy outBuffer -> readbackBuffer (global)
    // ensure readbackBuffer is large enough (we resized on upload)
    commandEncoder.copyBufferToBuffer(outBuffer, 0, this.readbackBuffer!, 0, outByteLength);

    const commands = commandEncoder.finish();
    this.device.queue.submit([commands]);

    // map readback buffer and read floats
    await this.readbackBuffer!.mapAsync(GPUMapMode.READ);
    const mappedRange = this.readbackBuffer!.getMappedRange(0, outByteLength);
    // slice makes a copy; mappedRange is an ArrayBuffer
    const scores = new Float32Array(mappedRange.slice(0));
    this.readbackBuffer!.unmap();

    // cleanup ephemeral outBuffer
    try {
      (outBuffer as any).destroy?.();
    } catch {}

    // update LRU timestamp
    shardRec.lastUsed = Date.now();

    return scores;
  }

  /**
   * Compute topK for a resident shard: uploads only query & params then reuses resident shard buffer.
   * Returns ShardTopK with local indices+scores (shardIndex filled by caller).
   */
  async computeTopKForResidentShard(
    shardIndex: number,
    query: Float32Array,
    dim: number,
    topK: number
  ): Promise<ShardTopK> {
    console.log(this.shardCache)
    const shardRec = this.shardCache.get(shardIndex);
    if (!shardRec) {
      throw new Error(`Shard ${shardIndex} not resident on GPU`);
    }

    // ensure query & params are written
    this.ensureQueryAndParams(dim, shardRec.count, query);

    // run compute that writes scores into global readback buffer
    const sims = await this.computeShardUsingResidentBuffer(shardRec, dim);

    // convert top-K in JS (you can later move Top-K to GPU)
    const count = shardRec.count;
    const scored: { idx: number; score: number }[] = new Array(count);
    for (let i = 0; i < count; i++) {
      scored[i] = { idx: i, score: sims[i] };
    }

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, Math.min(topK, scored.length));

    return {
      shardIndex,
      indices: top.map((t) => t.idx),
      scores: top.map((t) => t.score),
    };
  }

  // ---------------------------
  // Utility / housekeeping
  // ---------------------------

  /**
   * Get shard info for debugging
   */
  public getShardCacheInfo() {
    return {
      totalGpuBytes: this.totalGpuBytes,
      maxGpuBytes: this.maxGpuBytes,
      shards: Array.from(this.shardCache.values()).map((r) => ({
        shardIndex: r.shardIndex,
        count: r.count,
        startIndex: r.startIndex,
        byteSize: r.byteSize,
        lastUsed: r.lastUsed,
      })),
    };
  }

  /**
   * Destroy all GPU resources (call on app shutdown)
   */
  public destroyAll() {
    for (const rec of this.shardCache.values()) {
      try {
        (rec.gpuBuffer as any).destroy?.();
      } catch {}
    }
    this.shardCache.clear();
    this.totalGpuBytes = 0;

    try { (this.queryBuffer as any)?.destroy?.(); } catch {}
    try { (this.paramsBuffer as any)?.destroy?.(); } catch {}
    try { (this.readbackBuffer as any)?.destroy?.(); } catch {}
  }

  // ---------------------------
  // Example helper to integrate with your pipeline
  // ---------------------------

  /**
   * Convenience wrapper: ensure shard resident (upload if not), then compute TopK for that shard
   * - storeVectorsLoader should be a function that returns Float32Array for the shard (already provided in your StoreGithubPage)
   */
  async ensureResidentAndComputeShardTopK(
    shardIndex: number,
    storeVectorsLoader: (shardIndex: number) => Promise<Float32Array>,
    dim: number,
    topK: number,
    globalStartIndex: number
  ): Promise<ShardTopK> {
    // upload if missing
    if (!this.shardCache.has(shardIndex)) {
      const vectors = await storeVectorsLoader(shardIndex); // Float32Array
      const count = vectors.length / dim;
      await this.uploadShardToGPU(shardIndex, vectors, dim, count, globalStartIndex);
      // Optionally drop vectors reference in caller after this
    }

    // compute
    // Note: caller must provide the query through other method, e.g., computeAllShardsTopK will write query then call this
    throw new Error("This helper expects you to call computeTopKForResidentShard after writing query");
  }
}
