import { expect, test } from "vitest";
import { GpuSimilarityEngine } from "../gpu/gpuSimilarityEngineLatest";
import { StoreGithubPage } from "../stores/storeGithubPage";
import { buildAcronymRegex, vectorizeQuery } from "../tfIdf";
import { SearchResultItem } from "../pipeline";
import { ErrorWebGPUBuffer } from "../gpu/errors";

const store = new StoreGithubPage(
  "https://hariswb.github.io/indonesian-news-2024-2025"
);

await store.loadManifest();
await store.loadMetadataIndex();

const manifest = store.getManifest();
const dim = manifest.dim;
const maxShardCount = Math.max(...manifest.shards.map((shard) => shard.count));

const vocab = await store.loadVocab();
const idf = await store.loadIdf();
const stopwords = await store.loadStopwords();
const acronymDict = await store.loadAcronyms();

const gpu = new GpuSimilarityEngine(maxShardCount, dim);

gpu.init();

test("Pipeline Initial", async () => {
  await testGpuSimilarity("BBM naik hari ini apakah berdampak?", 10);
});

test("Pipeline 1", async () => {
  await testGpuSimilarity("Prabowo ekonomi", 10);
});

test("Pipeline 2", async () => {
  await testGpuSimilarity("Timnas indonesia", 10);
});

export async function testGpuSimilarity(query: string, topK: number) {
  const qvec = await getQVec(query);

  // Map<shard index, shard's start index> 
  const shardStartIdxMap: Map<number, number> = new Map();

  // Create buffer records
  let globalIdx = 0;
  for (let i = 0; i < manifest.shards.length; i++) {
    const shardInfo = manifest.shards[i];
    const vectors = await store.loadShard(i); // Float32Array
    await gpu.createBufferRecord(i, vectors, manifest.dim, shardInfo.count);
    shardStartIdxMap.set(i, globalIdx);
    globalIdx += shardInfo.count;
  }

  const globalResults: SearchResultItem[] = [];

  for (const shardRec of gpu.shardRecords.values()) {
    const shardScores = await gpu.computeTopKFromShard(
      shardRec.shardIndex,
      qvec,
      10
    );

    const shardStartIdx = shardStartIdxMap.get(shardRec.shardIndex);
    if (shardStartIdx) {
      for (const score of shardScores.scores) {
        globalResults.push({
          globalIndex: shardStartIdx + score.idx,
          score: score.score,
        });
      }
    }
  }
  globalResults.sort((a, b) => b.score - a.score);

  const titles = await store.fetchMetadataBatch(globalResults.slice(0,10).map((o) => o.globalIndex));

  console.log(titles.map(t=>t.title))
}

async function getQVec(query: string) {
  const acronymRegex = buildAcronymRegex(acronymDict);

  const qvec = vectorizeQuery(
    query,
    vocab,
    idf,
    acronymDict,
    acronymRegex,
    stopwords
  );

  return qvec;
}
