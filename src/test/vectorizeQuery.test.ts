import { expect, test } from "vitest";
import { StoreGithubPage} from "../stores/storeGithubPage";
import { buildAcronymRegex, vectorizeQuery } from "../utils/tfIdf";

test("Vectorize Query", async () => {
  const qvec = await testVectorizeQuery("BBM naik hari ini apakah berdampak?");

  expect(qvec).toBeInstanceOf(Float32Array)
  expect(qvec).toHaveLength(1000);
});

export async function testVectorizeQuery(query:string) {
  const store = new StoreGithubPage("https://hariswb.github.io/indonesian-news-2024-2025");

  const vocab = await store.loadVocab();
  const idf = await store.loadIdf();
  const stopwords = await store.loadStopwords();
  const acronymDict = await store.loadAcronyms();
  const acronymRegex = buildAcronymRegex(acronymDict);

  const qvec = vectorizeQuery(query, vocab, idf, acronymDict, acronymRegex, stopwords);

  return qvec
}
