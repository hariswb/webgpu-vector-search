// vectorSearchPipeline.bench.ts
import { beforeAll, afterAll, bench, expect } from "vitest";
import {
  DataIndexStream,
  ShardComputedScores,
  VectorScore,
  VectorSearchPipeline,
} from "../pipeline";

const pipeline = new VectorSearchPipeline(
  "https://hariswb.github.io/indonesian-news-2024-2025"
);

let shardComputedScores: ShardComputedScores[] | null = null;
let searchResult: VectorScore[] | null = null;

beforeAll(async () => {
  await pipeline.init();
});

bench("Compute", async () => {
  shardComputedScores = await pipeline.computeShardScores("Demonstrasi ricuh");
});

bench("Extract compute result", async () => {
  const globalResults: VectorScore[] = [];

  if (!shardComputedScores) throw new Error("Shard computed score null");

  // Stage 2 & 3 per shard — can be parallelized with Promise.all if desired
  for (const out of shardComputedScores) {
    const shardScores = await pipeline.extractShardScores(out);
    const global = await pipeline.reassignScoresToGlobalIndex(shardScores);
    globalResults.push(...global);
  }

  searchResult = globalResults.sort((a, b) => b.score - a.score);
});

bench("Fetch top 10 result", async () => {
  if (!searchResult) throw new Error("Search Result null");

  const streamDataIdx = new DataIndexStream(searchResult);

  const data = await pipeline.fetchData(streamDataIdx.next(10));
});

afterAll(async () => {
  await pipeline.cleanUp();
});
