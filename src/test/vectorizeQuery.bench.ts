import { test, bench } from "vitest";
import { StoreGithubPage } from "../stores/storeGithubPage";
import { QueryVectorizer } from "../utils/tfIdf";

const store = new StoreGithubPage(
  "https://hariswb.github.io/indonesian-news-2024-2025"
);

let queryVectorizer: QueryVectorizer | null = null;

test.beforeAll(async () => {
  const vocab = await store.loadVocab();
  const idf = await store.loadIdf();
  const stopwords = await store.loadStopwords();
  const acronymDict = await store.loadAcronyms();

  queryVectorizer = new QueryVectorizer(vocab, idf, acronymDict, stopwords);
});

bench("Vectorize", async()=>{
  if (queryVectorizer) {
    queryVectorizer.vectorize(
      "BBM naik hari ini apakah berdampak?"
    );
    return;
  }

  throw new Error("Query Vectorizer not available");

}, {iterations:10})
