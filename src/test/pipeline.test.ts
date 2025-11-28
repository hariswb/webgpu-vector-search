import { beforeAll, afterAll, test, expect, vi, it } from "vitest";
import { VectorSearchPipeline, DataIndexStream } from "../pipeline";

const topK = 10;

const pipeline = new VectorSearchPipeline(
  "https://hariswb.github.io/indonesian-news-2024-2025"
);

beforeAll(async () => {
  await pipeline.init();
});

test("Compute and out result", async () => {
  const sortedSearchResults = await pipeline.compute("Demonstrasi ricuh");

  expect(sortedSearchResults).toBeInstanceOf(Array);

  const streamDataIdx = new DataIndexStream(sortedSearchResults);

  const topIdx = streamDataIdx.next(topK);

  const data = await pipeline.getResults(topIdx);

  expect(data).toHaveLength(topK);

  for (let i = 0; i < topK; i++) {
    expect(data[i]).toHaveProperty("title");
  }
});

test("Compute and out result stream", async () => {
  const sortedSearchResults = await pipeline.compute("Keracunan MBG");

  expect(sortedSearchResults).toBeInstanceOf(Array);

  const streamDataIdx = new DataIndexStream(sortedSearchResults);

  const dataFirst = await pipeline.getResults(streamDataIdx.next(topK));
  expect(dataFirst).toHaveLength(topK);
  for (let i = 0; i < topK; i++) {
    expect(dataFirst[i]).toHaveProperty("title");
  }

  const dataSecond = await pipeline.getResults(streamDataIdx.next(topK));
  expect(dataSecond).toHaveLength(topK);
  for (let i = 0; i < topK; i++) {
    expect(dataSecond[i]).toHaveProperty("title");
  }
});

afterAll(async () => {
  await pipeline.cleanUp();
});
