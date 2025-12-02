import { beforeAll, afterAll, test, expect, vi, it } from "vitest";
import { VectorSearchPipeline, DataIndexStream } from "../pipeline";

const page = 10;

const pipeline = new VectorSearchPipeline(
  "https://hariswb.github.io/indonesian-news-2024-2025"
);

test("Fetch manifest and shards to load to buffers",async () => {
  await pipeline.init();
});

test("Compute and out result", async () => {
  const sortedSearchResults = await pipeline.search("Demonstrasi ricuh");

  expect(sortedSearchResults).toBeInstanceOf(Array);

  const streamDataIdx = new DataIndexStream(sortedSearchResults);

  const topIdx = streamDataIdx.next(page);

  const data = await pipeline.fetchData(topIdx);

  expect(data).toHaveLength(page);

  for (let i = 0; i < page; i++) {
    expect(data[i]).toHaveProperty("title");
  }
});

test("Compute and stream results", async () => {
  const sortedSearchResults = await pipeline.search("Keracunan MBG");

  expect(sortedSearchResults).toBeInstanceOf(Array);

  const streamDataIdx = new DataIndexStream(sortedSearchResults);

  const dataFirst = await pipeline.fetchData(streamDataIdx.next(page));
  expect(dataFirst).toHaveLength(page);
  for (let i = 0; i < page; i++) {
    expect(dataFirst[i]).toHaveProperty("title");
  }

  const dataSecond = await pipeline.fetchData(streamDataIdx.next(page));
  expect(dataSecond).toHaveLength(page);
  for (let i = 0; i < page; i++) {
    expect(dataSecond[i]).toHaveProperty("title");
  }
});

afterAll(async () => {
  await pipeline.cleanUp();
});
