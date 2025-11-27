import { useEffect, useRef, useState } from "react";
import { VectorSearchPipeline, DataIndexStream } from "../pipeline"; // adjust path

const PIPELINE_URL = "https://hariswb.github.io/indonesian-news-2024-2025"
const TOP_K = 200;

export default function App() {
  const pipelineRef = useRef<VectorSearchPipeline | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [isLoadingQuery, setIsLoadingQuery] = useState(false);

  const [stream, setStream] = useState<DataIndexStream | null>(null);
  const [results, setResults] = useState<any[]>([]); // metadata objects

  // -------------------------
  // 1. INIT PIPELINE ON MOUNT
  // -------------------------
  useEffect(() => {
    const pipeline = new VectorSearchPipeline(PIPELINE_URL, TOP_K);
    pipelineRef.current = pipeline;

    pipeline
      .init()
      .then(() => {
        const checkReady = () => {
          if (pipeline.isReady()) setIsReady(true);
          else setTimeout(checkReady, 300);
        };
        checkReady();
      })
      .catch((e) => console.error("Pipeline init error:", e));
  }, []);

  // -------------------------
  // 2. DEBOUNCE QUERY (1 sec)
  // -------------------------
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), 1000);
    return () => clearTimeout(t);
  }, [query]);

  // -------------------------
  // 3. RUN COMPUTE WHEN DEBOUNCED QUERY CHANGES
  // -------------------------
  useEffect(() => {
    const run = async () => {
      if (!debouncedQuery || !pipelineRef.current) {
        setStream(null);
        setResults([]);
        return;
      }

      setIsLoadingQuery(true);

      const pipeline = pipelineRef.current;

      try {
        const globalIdxList = await pipeline.compute(debouncedQuery);
        const newStream = new DataIndexStream(globalIdxList);

        // take first 10 indexes
        const firstChunk = newStream.next(10);

        const firstMetadata = await pipeline.getResults(firstChunk);

        setStream(newStream);
        setResults(firstMetadata);
      } catch (err) {
        console.error("Error computing:", err);
      }

      setIsLoadingQuery(false);
    };

    run();
  }, [debouncedQuery]);

  // -------------------------
  // 4. LOAD MORE RESULTS
  // -------------------------
  const loadMore = async () => {
    if (!stream || !pipelineRef.current) return;
    if (!stream.hasMore()) return;

    const pipeline = pipelineRef.current;
    const nextIdx = stream.next(10);
    const moreMetadata = await pipeline.getResults(nextIdx);

    setResults((prev) => [...prev, ...moreMetadata]);
  };

  // -------------------------
  // UI RENDERING
  // -------------------------

  if (!isReady) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-700 text-xl">
        Initializing WebGPU pipeline...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 p-8">
      <div className="max-w-3xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold">WebGPU Vector Search</h1>

        {/* Search Bar */}
        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Search article titles..."
            className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        {isLoadingQuery && (
          <div className="text-gray-500 italic mt-4">Computing similarity...</div>
        )}

        {/* Empty */}
        {!query.trim() && (
          <div className="p-4 text-center text-gray-500 italic">
            Find news with WebGPU
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-3">
            {results.map((item, i) => (
              <div key={i} className="p-4 bg-white rounded-xl shadow">
                <div className="font-semibold">{item.title}</div>
                <div className="text-sm text-gray-600">{item.date}</div>
                <a
                  href={item.url}
                  target="_blank"
                  className="text-blue-600 underline text-sm"
                >
                  View article
                </a>
              </div>
            ))}

            {stream && stream.hasMore() && (
              <button
                className="w-full p-3 bg-blue-600 text-white rounded-lg text-center mt-4"
                onClick={loadMore}
              >
                Load more results
              </button>
            )}
          </div>
        )}

        {!isLoadingQuery && query.trim() && results.length === 0 && (
          <div className="p-4 text-center text-gray-500">
            No results found.
          </div>
        )}
      </div>
    </div>
  );
}
