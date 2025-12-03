import {
  ErrorWebGPUBuffer,
  ErrorWebGPUCompute,
  ErrorWebGPUInit,
  ErrorWebGPUNotSupported,
} from "./errors";

import { ShardRecord, VecScore, ShardScores } from "./types";

export class GpuSimilarityEngine {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private paramsBuffer: GPUBuffer | null = null;
  private queryBuffer: GPUBuffer | null = null;
  private readbackBuffer: GPUBuffer | null = null;
  public shardRecords: Map<number, ShardRecord> = new Map();

  private vecDimension: number | null = null;
  private maxShardCount: number | null = null;

  constructor(maxShardCount: number, dim: number) {
    this.vecDimension = dim;
    this.maxShardCount = maxShardCount;
  }

  async init(): Promise<void> {
    if (
      !(
        typeof navigator !== "undefined" &&
        "gpu" in navigator &&
        navigator.gpu !== undefined
      )
    ) {
      throw new ErrorWebGPUNotSupported(
        "WebGPU not supported in this browser."
      );
    }

    this.adapter = await navigator.gpu.requestAdapter();

    if (!this.adapter)
      throw new ErrorWebGPUNotSupported("Failed to request GPU adapter.");

    this.device = await this.adapter.requestDevice();
    if (!this.device)
      throw new ErrorWebGPUNotSupported("need a browser that supports WebGPU");

    if (!this.vecDimension)
      throw new ErrorWebGPUInit("Shards' vector count is null");
    if (!this.maxShardCount)
      throw new ErrorWebGPUInit("Shards' vector dimension is null");

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
      label: "Bind group layout compute",
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

    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    this.paramsBuffer = this.device.createBuffer({
      label: "Param buffer",
      size: 8, //two u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.queryBuffer = this.device.createBuffer({
      label: "Query buffer",
      size: this.vecDimension * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.readbackBuffer = this.device.createBuffer({
      label: "Readback buffer",
      size: this.maxShardCount * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  async createBufferRecord(
    shardIndex: number,
    shard: Float32Array,
    dim: number,
    count: number
  ): Promise<void> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new ErrorWebGPUInit("Engine not initialized. Call init() first.");
    }

    if (shard.length !== dim * count) {
      throw new ErrorWebGPUBuffer(
        `shard length (${shard.length}) != dim*count (${dim * count})`
      );
    }

    const byteLength = shard.byteLength;

    const gpuBuffer = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    try {
      const mappedRange = gpuBuffer.getMappedRange();
      const mappedF32 = new Float32Array(mappedRange as ArrayBuffer);
      mappedF32.set(shard);
      gpuBuffer.unmap();
    } catch (e) {
      if (e instanceof GPUOutOfMemoryError) {
        throw new ErrorWebGPUBuffer("Web GPU out of memory");
      } else {
        throw e;
      }
    }

    const record: ShardRecord = {
      shardIndex: shardIndex,
      byteSize: byteLength,
      count: count,
      dim: dim,
      gpuBuffer: gpuBuffer,
    };
    this.shardRecords.set(shardIndex, record);
  }

  async computeShard(
    shardRec: ShardRecord,
    query: Float32Array
  ): Promise<Float32Array> {
    if (!this.vecDimension)
      throw new ErrorWebGPUInit("Shards' vector count is null");
    if (!this.maxShardCount)
      throw new ErrorWebGPUInit("Shards' vector dimension is null");

    if (
      !this.device ||
      !this.pipeline ||
      !this.bindGroupLayout ||
      !this.queryBuffer ||
      !this.paramsBuffer ||
      !this.readbackBuffer
    ) {
      throw new ErrorWebGPUInit("Engine not initialized.");
    }

    if (!this.isQueryValid(query)) {
      throw new ErrorWebGPUCompute("Query is invalid.");
    }

    if (!this.isParamValid(shardRec)) {
      throw new ErrorWebGPUCompute("Param is invalid.");
    }

    // Write query byte to queryBuffer
    this.device.queue.writeBuffer(
      this.queryBuffer,
      0,
      query.buffer,
      query.byteOffset,
      query.length * 4
    );

    // Write params
    const paramsArray = new Uint32Array([
      this.vecDimension >>> 0,
      shardRec.count >>> 0,
    ]);
    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      paramsArray.buffer,
      paramsArray.byteOffset,
      paramsArray.byteLength
    );

    // Create outbuffer to store before readback

    const outBuffer = this.device.createBuffer({
      label: "Out buffer",
      size: this.maxShardCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Bind group
    const bindGroup = this.device.createBindGroup({
      label: `Bind group ${shardRec.shardIndex}`,
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.queryBuffer } },
        { binding: 2, resource: { buffer: shardRec.gpuBuffer } },
        { binding: 3, resource: { buffer: outBuffer } },
      ],
    });

    // Command Buffer

    const commandEncoder = this.device.createCommandEncoder({
      label: `Compute shard ${shardRec.shardIndex}`,
    });

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    const workgroupSize = 64;
    const workgroupCount = Math.ceil(shardRec.count / workgroupSize);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    //  Read back out buffer

    commandEncoder.copyBufferToBuffer(
      outBuffer,
      0,
      this.readbackBuffer,
      0,
      this.maxShardCount * 4
    );

    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const mappedRange = this.readbackBuffer.getMappedRange();

    const scores = new Float32Array(mappedRange.slice(0));
    this.readbackBuffer.unmap();

    // Cleanup ephemeral outBuffer
    outBuffer.destroy();

    return scores;
  }

  async destroyBuffers() {
    for (const [key, record] of this.shardRecords.entries()) {
      try {
        record.gpuBuffer.destroy();
        this.shardRecords.delete(key);
      } catch (e) {
        throw new ErrorWebGPUBuffer(
          `Fail to destroy shard buffer ${record.shardIndex}.
          Message: ${(e as Error).message}`
        );
      }
    }

    try {
      this.queryBuffer?.destroy();
    } catch (e) {
      throw new ErrorWebGPUBuffer(
        `Fail to destroy query buffer.
          Message: ${(e as Error).message}`
      );
    }

    try {
      this.paramsBuffer?.destroy();
    } catch (e) {
      throw new ErrorWebGPUBuffer(
        `Fail to destroy param buffer.
          Message: ${(e as Error).message}`
      );
    }

    try {
      this.readbackBuffer?.destroy();
    } catch (e) {
      throw new ErrorWebGPUBuffer(
        `Fail to destroy out buffer.
          Message: ${(e as Error).message}`
      );
    }
  }

  private isQueryValid(query: Float32Array) {
    return query.length === this.vecDimension;
  }

  private isParamValid(shard: ShardRecord) {
    if (this.vecDimension === null) return false;
    return shard.dim === this.vecDimension;
  }
}
