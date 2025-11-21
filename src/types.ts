// types.ts (or inline)
type ShardEntry = {
  shard: string;
  url: string;
  start_index: number;
  count: number;
  size_bytes: number;
};

type Manifest = {
  total_vectors: number;
  dim: number;
  dtype: string;
  header_size: number;
  shards: ShardEntry[];
};
