/**
 * GpuSimilarityEngine:
 * - init(): obtains adapter/device and creates compute pipeline
 * - computeShard(): runs shader for a single shard and returns scores (Float32Array)
 * - getTopK(): uses a heap-based top-K to return top results across shards
 *
 * Usage:
 *   const eng = new GpuSimilarityEngine();
 *   await eng.init();
 *   const scores = await eng.computeShard(queryVec, shardFloat32, dim, count);
 *   eng.addShardResult(shardStartIndex, scores, shardId);
 *   const top = eng.getTopK(50);
 */

import { InternalEntry, ShardTopK, TopKResult } from "./types";

export class GpuSimilarityEngine {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  // collected results if you want to merge incrementally
  private entries: InternalEntry[] = [];

  constructor() {}

  /** Initialize WebGPU: adapter, device, pipeline */
  async init(): Promise<void> {
    if (!("gpu" in navigator)) {
      throw new Error("WebGPU not supported in this browser.");
    }
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) throw new Error("Failed to request GPU adapter.");

    this.device = await this.adapter.requestDevice();

    if (!this.device) throw new Error("need a browser that supports WebGPU");
    
    // Create shader module & pipeline
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
        // params uniform
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        // query buffer (storage)
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        // vectors buffer (storage)
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        // out buffer (storage read_write)
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
  }

  /**
   * Compute dot product scores for a shard.
   *
   * @param query Float32Array (length == dim)
   * @param shardVectors Float32Array (length == dim * count)
   * @param dim number of dimensions
   * @param count number of vectors in shard
   * @returns Float32Array of length count with dot products (cosine if vectors are normalized)
   */
  async computeShard(
    query: Float32Array,
    shardVectors: Float32Array,
    dim: number,
    count: number,
  ): Promise<Float32Array> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error("Engine not initialized. Call init() first.");
    }
    if (query.length !== dim) {
      throw new Error(`query length (${query.length}) != dim (${dim})`);
    }
    if (shardVectors.length !== dim * count) {
      throw new Error(
        `shardVectors length (${shardVectors.length}) != dim*count (${
          dim * count
        })`
      );
    }

    const device = this.device;

    // --- create uniform buffer for params (dim, count) (8 bytes) ---
    const uniformBufSize = 8; // two u32
    const uniformBuffer = device.createBuffer({
      size: uniformBufSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // write uniform values into an ArrayBuffer
    const uniformArray = new Uint32Array([dim >>> 0, count >>> 0]);
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      uniformArray.buffer,
      uniformArray.byteOffset,
      uniformArray.byteLength
    );

    // --- create & upload query buffer (storage read-only) ---
    const queryByteLength = query.length * 4;
    const queryBuffer = device.createBuffer({
      size: queryByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      queryBuffer,
      0,
      query.buffer,
      query.byteOffset,
      queryByteLength
    );

    // --- create & upload vectors buffer (storage read-only) ---
    const vecByteLength = shardVectors.length * 4;
    const vectorsBuffer = device.createBuffer({
      size: vecByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      vectorsBuffer,
      0,
      shardVectors.buffer,
      shardVectors.byteOffset,
      vecByteLength
    );

    // --- create output storage buffer (storage, copy-src) ---
    const outByteLength = count * 4;
    const outBuffer = device.createBuffer({
      size: outByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // --- bind group ---
    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: queryBuffer } },
        { binding: 2, resource: { buffer: vectorsBuffer } },
        { binding: 3, resource: { buffer: outBuffer } },
      ],
    });

    // --- command encoder & dispatch ---
    const commandEncoder = device.createCommandEncoder();

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    // calculate dispatch size: ceil(count / 64)
    const workgroupSize = 64;
    const dispatchX = Math.ceil(count / workgroupSize);
    pass.dispatchWorkgroups(dispatchX);
    pass.end();

    // create a readback buffer and copy outBuffer -> readBuffer
    const readBuffer = device.createBuffer({
      size: outByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    commandEncoder.copyBufferToBuffer(
      outBuffer,
      0,
      readBuffer,
      0,
      outByteLength
    );

    // submit
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // map readBuffer
    await readBuffer.mapAsync(GPUMapMode.READ);
    const mappedRange = readBuffer.getMappedRange();
    // copy bytes to a Float32Array
    const scores = new Float32Array(mappedRange.slice(0));
    readBuffer.unmap();

    // cleanup (destroy buffers if available / desirable)
    // Note: Some browsers implement .destroy() but it's optional; letting GC handle it is fine.
    return scores;
  }

  async computeTopKForShard(
    query: Float32Array,
    vectors: Float32Array,
    dim: number,
    vecCount: number,
    topK: number
  ): Promise<ShardTopK> {
    const sims = await this.computeShard(query, vectors, dim, vecCount);

    // compute similarities for all rows in this shard
    const count = vectors.length / dim;

    const scored: { idx: number; score: number }[] = [];

    for (let i = 0; i < count; i++) {
      scored.push({
        idx: i,
        score: sims[i],
      });
    }

    scored.sort((a, b) => b.score - a.score);

    const top = scored.slice(0, topK);

    return {
      shardIndex: -1, // will be filled in the pipeline
      indices: top.map((t) => t.idx),
      scores: top.map((t) => t.score),
    };
  }

}
