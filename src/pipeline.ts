import { MetadataObject, StoreGithubPage } from "./stores/storeGithubPage";
import { GpuSimilarityEngine } from "./gpu/engine";
import { QueryVectorizer } from "./utils/tfIdf";
import { ShardRecord, ShardScores, VecScore } from "./gpu/types";

export interface ShardComputedScores {
  shardIndex: number;
  rawScores: Float32Array;
  count: number;
}

export interface VectorScore {
  globalIndex: number;
  score: number;
}

export class VectorSearchPipeline {
  private storeUrl: string;

  private store: StoreGithubPage;
  private gpu: GpuSimilarityEngine | null = null;
  private queryVectorizer: QueryVectorizer | null = null;

  private shardStartIdxMap: Map<ShardRecord["shardIndex"], number>;

  constructor(storeUrl: string) {
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
    // Keep shard starting index
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

  async search(query: string): Promise<VectorScore[]> {
    // Stage 1 — Compute GPU in parallel
    const shardOutputs = await this.computeShardScores(query);

    const globalResults: VectorScore[] = [];

    // Stage 2 & 3 per shard — can be parallelized with Promise.all if desired
    for (const out of shardOutputs) {
      const shardScores = await this.extractShardScores(out);
      const global = await this.reassignScoresToGlobalIndex(shardScores);
      globalResults.push(...global);
    }

    return globalResults.sort((a, b) => b.score - a.score);
  }

  async computeShardScores(query: string): Promise<ShardComputedScores[]> {
    if (!this.gpu || !this.queryVectorizer)
      throw new Error("Pipeline not initialized");

    const qvec = this.queryVectorizer.vectorize(query);
    const outputs: ShardComputedScores[] = [];

    for (const [shardIndex, shardRec] of this.gpu.shardRecords.entries()) {
      const rawScores = await this.gpu.computeShard(shardRec, qvec);
      outputs.push({
        shardIndex,
        rawScores,
        count: shardRec.count,
      });
    }

    return outputs;
  }

  async extractShardScores(input: ShardComputedScores): Promise<ShardScores> {
    const { shardIndex, rawScores, count } = input;

    const scored: VecScore[] = new Array(count);
    for (let i = 0; i < count; i++) {
      scored[i] = { idx: i, score: rawScores[i] };
    }

    scored.sort((a, b) => b.score - a.score);

    return {
      shardIdx: shardIndex,
      scores: scored,
    };
  }

  async reassignScoresToGlobalIndex(
    shardResult: ShardScores
  ): Promise<VectorScore[]> {
    const start = this.shardStartIdxMap.get(shardResult.shardIdx) ?? 0;
    return shardResult.scores.map((it) => ({
      globalIndex: start + it.idx,
      score: it.score,
    }));
  }

  async fetchData(vectorScores: VectorScore[]): Promise<MetadataObject[]> {
    const titles = await this.store.fetchMetadataBatch(vectorScores.map(v=>v.globalIndex));
    return titles;
  }

  async cleanUp() {
    await this.gpu?.destroyBuffers();
    this.gpu = null;
  }
}

export class DataIndexStream {
  private index = 0;

  constructor(private data: VectorScore[]) {}

  next(k: number): VectorScore[] {
    const slice = this.data.slice(this.index, this.index + k);
    this.index += slice.length; // do NOT exceed bounds
    return slice;
  }

  hasMore(): boolean {
    return this.index < this.data.length;
  }
}
