import { StoreGithubPage } from "./stores/storeGithubPage";
import { GpuSimilarityEngine } from "./gpu/gpuSimilarityEngine";

export interface SearchResultItem {
  globalIndex: number;
  score: number;
}

export class VectorSearchPipeline {
  private store: StoreGithubPage;
  private gpu: GpuSimilarityEngine;

  constructor(store: StoreGithubPage, gpu: GpuSimilarityEngine) {
    this.store = store;
    this.gpu = gpu;
  }

  async computeAllShardsTopK(
    query: Float32Array,
    topK: number
  ): Promise<SearchResultItem[]> {
    const manifest = this.store.getManifest();
    const dim = manifest.dim;

    const globalResults: SearchResultItem[] = [];

    for (let s = 0; s < manifest.shards.length; s++) {
      console.log(`Processing shard ${s}...`);

      const shardInfo = manifest.shards[s];

      // Load shard vectors
      const vectors = await this.store.loadShard(s);

      // Compute per-shard top-K
      const shardTop = await this.gpu.computeTopKForShard(
        query,
        vectors,
        dim,
        shardInfo.count,
        topK
      );

      // Fill shardIndex
      shardTop.shardIndex = s;

      // Convert local shard index -> global vector index
      for (let i = 0; i < shardTop.indices.length; i++) {
        const localIdx = shardTop.indices[i];
        const score = shardTop.scores[i];

        const globalIdx = shardInfo.start_index + localIdx;

        globalResults.push({
          globalIndex: globalIdx,
          score,
        });
      }
    }

    // Merge across all shards
    globalResults.sort((a, b) => b.score - a.score);

    return globalResults.slice(0, topK);
  }
}
