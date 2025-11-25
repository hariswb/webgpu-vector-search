export interface ShardRecord {
  shardIndex: number;
  byteSize: number;
  count: number;
  dim: number;
  gpuBuffer: GPUBuffer;
}

export interface VecScore {
  idx: number;
  score: number;
}

export interface ShardScores {
  shardIdx: number;
  scores: VecScore[];
}