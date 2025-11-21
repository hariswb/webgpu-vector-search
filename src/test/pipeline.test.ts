import { expect, test } from "vitest";
import { GpuSimilarityEngine } from "../gpu/gpuSimilarityEngine";
import { VectorSearchPipeline } from "../pipeline";
import { StoreGithubPage } from "../stores/storeGithubPage";
import { buildAcronymRegex, vectorizeQuery } from "../tfIdf";

const store = new StoreGithubPage(
  "https://hariswb.github.io/indonesian-news-2024-2025"
);

await store.loadManifest();
await store.loadMetadataIndex();

const vocab = await store.loadVocab();
const idf = await store.loadIdf();
const stopwords = await store.loadStopwords();
const acronymDict = await store.loadAcronyms();

const gpu = new GpuSimilarityEngine();
gpu.init();
const pipeline = new VectorSearchPipeline(store, gpu);

test("Is WebGPU Available", async () => {
  expect("gpu" in navigator).toBe(true);
});

test("Pipeline First Load", async () => {
  expect(await testGpuSimilarity("BBM naik hari ini apakah berdampak?",10)).toBe(10);
});

test("Pipeline Cached", async () => {
  expect(await testGpuSimilarity("Kebijakan prabowo",10)).toBe(10);
});

export async function testGpuSimilarity(query:string,topK:number) {
  console.time(`Elapsed - ${query}`)
  const qvec = await getQVec(query);

  const top = await pipeline.computeAllShardsTopK(qvec, topK);

  const results = await store.fetchMetadataBatch(top.map((o) => o.globalIndex));

  console.timeEnd(`Elapsed - ${query}`)
  return results.length
}

async function getQVec(query:string) {
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
