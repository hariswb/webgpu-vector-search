export type InternalEntry = { index: number; score: number; shardId?: number };
export type TopKResult = { index: number; score: number; shardId?: number };
export interface ShardTopK {
  shardIndex: number;
  indices: number[];         // local indices inside shard
  scores: number[];          // cosine scores
}
