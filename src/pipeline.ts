import { MetadataObject, StoreGithubPage } from "./stores/storeGithubPage";
import { GpuSimilarityEngine } from "./gpu/engine";
import { QueryVectorizer } from "./utils/tfIdf";
import { ShardRecord } from "./gpu/types";

export interface SortedSearchResult {
  globalIndex: number;
  score: number;
}

export class VectorSearchPipeline {
  private resultK: number;
  private storeUrl: string;

  private store: StoreGithubPage;
  private gpu: GpuSimilarityEngine | null = null;
  private queryVectorizer: QueryVectorizer | null = null;

  private shardStartIdxMap: Map<ShardRecord["shardIndex"], number>;

  constructor(storeUrl: string, resultK: number) {
    this.resultK = resultK;
    this.storeUrl = storeUrl;

    this.store = new StoreGithubPage(this.storeUrl);

    this.shardStartIdxMap = new Map();
  }

  async init() {
    // Init Store
    await this.store.loadManifest();
    await this.store.loadMetadataIndex();
    const manifest = this.store.getManifest();

    const dim = manifest.dim;
    const maxShardCount = Math.max(
      ...manifest.shards.map((shard) => shard.count)
    );

    // Init WebGPU Engine
    this.gpu = new GpuSimilarityEngine(maxShardCount, dim);
    await this.gpu.init();

    // Create buffer records
    let globalIdx = 0;
    for (let i = 0; i < manifest.shards.length; i++) {
      const shardInfo = manifest.shards[i];
      const vectors = await this.store.loadShard(i); // Float32Array
      await this.gpu.createBufferRecord(
        i,
        vectors,
        manifest.dim,
        shardInfo.count
      );
      this.shardStartIdxMap.set(i, globalIdx);
      globalIdx += shardInfo.count;
    }

    // Init Query Vectorizer
    const vocab = await this.store.loadVocab();
    const idf = await this.store.loadIdf();
    const stopwords = await this.store.loadStopwords();
    const acronymDict = await this.store.loadAcronyms();

    this.queryVectorizer = new QueryVectorizer(
      vocab,
      idf,
      acronymDict,
      stopwords
    );
  }

  isReady(): boolean {
    try {
      if (!this.store.getManifest()) return false;
      if (!this.gpu?.shardRecords) return false;
      if (!this.queryVectorizer) return false;
      return true;
    } catch (e) {
      return false;
    }
  }

  async getResults(sortedSearchIdx: number[]): Promise<MetadataObject[]> {
    const titles = await this.store.fetchMetadataBatch(sortedSearchIdx);
    return titles;
  }

  async compute(query: string): Promise<number[]> {
    if (!this.gpu || !this.store || !this.queryVectorizer) {
      throw new Error("Pipeline not initialized");
    }

    const qvec = this.queryVectorizer.vectorize(query);

    const globalResults: SortedSearchResult[] = [];

    for (let i = 0; i < this.gpu.shardRecords.size; i++) {
      const shardRec = this.gpu.shardRecords.get(i);

      if (!shardRec) break;

      const shardScores = await this.gpu.computeTopKFromShard(
        shardRec.shardIndex,
        qvec,
        shardRec.count
      );

      const shardStartIdx = this.shardStartIdxMap.get(shardRec.shardIndex);
      if (shardStartIdx) {
        for (const score of shardScores.scores) {
          globalResults.push({
            globalIndex: shardStartIdx + score.idx,
            score: score.score,
          });
        }
      }
    }

    return globalResults
      .sort((a, b) => b.score - a.score)
      .map((o) => o.globalIndex);
  }

  async cleanUp(){
    await this.gpu?.destroyBuffers()
    this.gpu = null
  }
}

export class DataIndexStream {
  private index = 0;

  constructor(private data: number[]) {}

  next(k: number): number[] {
    const slice = this.data.slice(this.index, this.index + k);
    this.index += slice.length; // do NOT exceed bounds
    return slice;
  }

  hasMore(): boolean {
    return this.index < this.data.length;
  }
}
