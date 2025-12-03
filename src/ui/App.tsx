import { useEffect, useRef, useState } from "react";
import { VectorSearchPipeline, DataIndexStream } from "../pipeline"; // adjust path
import LoadingCircle from "./components/loading";

const PIPELINE_URL = "https://hariswb.github.io/indonesian-news-2024-2025";

export default function App() {
  const pipelineRef = useRef<VectorSearchPipeline | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [isLoadingQuery, setIsLoadingQuery] = useState(false);

  const [stream, setStream] = useState<DataIndexStream | null>(null);
  const [results, setResults] = useState<any[]>([]); // metadata objects

  // 1. INIT PIPELINE ON MOUNT
  useEffect(() => {
    const pipeline = new VectorSearchPipeline(PIPELINE_URL);
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

  // 2. DEBOUNCE QUERY
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), 500);
    return () => clearTimeout(t);
  }, [query]);

  // 3. RUN COMPUTE WHEN DEBOUNCED QUERY CHANGES
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
        const globalIdxList = await pipeline.search(debouncedQuery);
        const newStream = new DataIndexStream(globalIdxList);

        // take first 10 indexes
        const firstChunk = newStream.next(10);

        const firstMetadata = await pipeline.fetchData(firstChunk);

        setStream(newStream);
        setResults(firstMetadata);
      } catch (err) {
        console.error("Error computing:", err);
      }

      setIsLoadingQuery(false);
    };

    run();
  }, [debouncedQuery]);

  // 4. LOAD MORE RESULTS
  const loadMore = async () => {
    if (!stream || !pipelineRef.current) return;
    if (!stream.hasMore()) return;

    const pipeline = pipelineRef.current;
    const nextIdx = stream.next(10);
    const moreMetadata = await pipeline.fetchData(nextIdx);

    setResults((prev) => [...prev, ...moreMetadata]);
  };

  // UI RENDERING

  if (!isReady) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-gray-700 text-xl">
        <div>Initializing WebGPU pipeline...</div>
        <div>
          <LoadingCircle />
        </div>
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
            placeholder="Search news super fast..."
            className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        {isLoadingQuery && (
          <div className="flex flex-col justify-center">
            <div className="text-gray-500 italic mt-4">
              Computing similarity...
            </div>
            <LoadingCircle />
          </div>
        )}

        {/* Empty */}
        {!query.trim() && (
          <div className="flex flex-col justify-center p-4 text-gray-500 italic">
            <p className="text-left ">
              Search within 90k Indonesian news dataset<br></br>
              between January 2024 - 23 October 2025 <br></br>
              <br></br>
              Example query:<br></br>
              1. Mitigasi bencana <br></br>
              2. Minyak goreng <br></br>
              3. Demonstrasi rusuh
            </p>
            <br></br>
            <p>
              <a
                target="_blank"
                href="https://github.com/hariswb/webgpu-vector-search"
              >
                <span className="text-blue-400 underline">
              Github Repository</span>
              </a>
            </p>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-3">
            {results.map((item, i) => (
              <div>
              <a
                href={item.url}
                target="_blank"
                className="cursor-pointer"
              >
                <div key={i} className="p-4 bg-white rounded-md border border-blue-200 hover:border-blue-400 hover:bg-blue-50">
                  <div className="font-semibold">{item.title}</div>
                  <div className="text-sm text-gray-600">{item.date}</div>
                </div>
              </a>
</div>
            ))}

            {stream && stream.hasMore() && (
              <button
                className="w-full p-3 bg-sky-600 text-white rounded-lg text-center mt-4 hover:cursor-pointer hover:bg-sky-800"
                onClick={loadMore}
              >
                Load more results
              </button>
            )}
          </div>
        )}

        {!isLoadingQuery && query.trim() && results.length === 0 && (
          <div className="p-4 text-center text-gray-500">No results found.</div>
        )}
      </div>
    </div>
  );
}
