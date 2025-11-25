import { expect, test } from "vitest";
import { GpuSimilarityEngine } from "../gpu/gpuSimilarityEngineNew";
import { StoreGithubPage } from "../stores/storeGithubPage";
import { buildAcronymRegex, vectorizeQuery } from "../tfIdf";
import { SearchResultItem } from "../pipeline";

const store = new StoreGithubPage(
  "https://hariswb.github.io/indonesian-news-2024-2025"
);

await store.loadManifest();
await store.loadMetadataIndex();

const manifest = store.getManifest();

const vocab = await store.loadVocab();
const idf = await store.loadIdf();
const stopwords = await store.loadStopwords();
const acronymDict = await store.loadAcronyms();

const gpu = new GpuSimilarityEngine();

gpu.init();

test("Pipeline New", async () => {
  const dim = manifest.dim;

  await testGpuSimilarity("BBM naik hari ini apakah berdampak?", 10);
});

export async function testGpuSimilarity(query: string, topK: number) {
  console.time(`Elapsed - ${query}`);
  const qvec = await getQVec(query);

  for (let s = 0; s < manifest.shards.length; s++) {
    const shardInfo = manifest.shards[s];
    const vectors = await store.loadShard(s); // Float32Array
    await gpu.uploadShardToGPU(
      s,
      vectors,
      manifest.dim,
      shardInfo.count,
      shardInfo.start_index
    );
    // optionally release vectors from JS after upload
    // vectors = null;
  }

  gpu["ensureQueryAndParams"](
    manifest.dim,
    /*count doesn't matter here for param write*/ 0,
    qvec
  ); // or make it public properly

  const globalResults: SearchResultItem[] = [];
  for (let s = 0; s < manifest.shards.length; s++) {
    const shardInfo = manifest.shards[s];
    const shardTop = await gpu.computeTopKForResidentShard(
      s,
      qvec,
      manifest.dim,
      topK
    );
    // convert local idx -> global index
    for (let i = 0; i < shardTop.indices.length; i++) {
      globalResults.push({
        globalIndex: shardInfo.start_index + shardTop.indices[i],
        score: shardTop.scores[i],
      });
    }
  }
  globalResults.sort((a, b) => b.score - a.score);

  console.log(globalResults);
  console.timeEnd(`Elapsed - ${query}`);

  return globalResults.slice(0, topK);
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
