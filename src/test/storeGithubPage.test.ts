import { expect, test } from "vitest";
import { StoreGithubPage} from "../stores/storeGithubPage";

const store = new StoreGithubPage("https://hariswb.github.io/indonesian-news-2024-2025");
await store.loadManifest();
const manifest = store.getManifest();

test("Store Github Manifest", async () => {
  expect(manifest).toHaveProperty("total_vectors")
  expect(manifest).toHaveProperty("dim", 1000)
  expect(manifest).toHaveProperty("dtype", "float32")
  expect(manifest).toHaveProperty("shards")
  expect(manifest.shards).toHaveLength(10);
});

test("Store Github Metadata", async ()=>{
  await store.loadMetadataIndex();

  const meta = await store.fetchMetadataLine(0);

  expect(meta).toHaveProperty("date")
  expect(meta).toHaveProperty("title")
  expect(meta).toHaveProperty("url")
})

test("Store Github Load Shard", async ()=> {
  const vectors = await store.loadShard(0);

  expect(vectors).toBeInstanceOf(Float32Array)
  expect(vectors).toHaveLength(manifest.dim * manifest.shards[0].count)
})
