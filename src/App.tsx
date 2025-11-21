import { useState } from "react";

const dummyArticles = [
  "Indonesia eyes renewable energy expansion in 2025",
  "Jakarta traffic improves after new regulation",
  "Tech startups in Bandung attract foreign investors",
  "Farmers celebrate strong harvest season",
  "New policies aim to boost national tourism",
  "Inflation slows as food prices stabilize",
  "Experts warn of rising cybersecurity threats",
  "Local universities collaborate on AI research",
  "Government announces digital ID initiative",
  "Online shopping growth continues in Southeast Asia",
];

export default function App() {
  const [query, setQuery] = useState("");

  const filtered = dummyArticles.filter((t) =>
    t.toLowerCase().includes(query.toLowerCase())
  );

  const isEmpty = query.trim() === "";

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

        {/* Empty state */}
        {isEmpty && (
          <div className="p-4 text-center text-gray-500 italic">
            Find news with WebGPU
          </div>
        )}

        {/* Results */}
        {!isEmpty && (
          <div className="space-y-3">
            {filtered.map((title, i) => {
              const lower = title.toLowerCase();
              const idx = lower.indexOf(query.toLowerCase());
              const before = title.slice(0, idx);
              const match = title.slice(idx, idx + query.length);
              const after = title.slice(idx + query.length);

              return (
                <div
                  key={i}
                  className="p-4 bg-white rounded-xl shadow"
                >
                  <span>{before}</span>
                  <span className="bg-yellow-200 font-semibold">{match}</span>
                  <span>{after}</span>
                </div>
              );
            })}

            {filtered.length === 0 && (
              <div className="p-4 text-center text-gray-500">
                No results found.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
