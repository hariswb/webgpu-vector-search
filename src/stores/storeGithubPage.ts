export interface EmbeddingShard {
  id: number;
  shard: string;
  url: string;
  start_index: number;
  count: number;
  shard_bytes: number;
}

export interface VectorManifest {
  dim: number;
  total_vectors: number;
  shards: EmbeddingShard[];
}

export interface MetadataObject {
  title: string;
  content: string;
  url: string;
  date: string; // or Date, depending on your JSONL
}

export class StoreGithubPage {
  private baseUrl: string;

  // Vector data
  private manifest: VectorManifest | null = null;
  private shardCache = new Map<number, Float32Array>(); // cached vectors

  // Metadata index
  private metadataOffsets: number[] | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  // --------------------------------------------------
  // Stop words and acronym
  // --------------------------------------------------

  async loadStopwords(): Promise<Set<string>> {
    const res = await fetch(this.baseUrl + "/stopword-id.csv");
    const text = await res.text();
    const words = text.split(/\r?\n/).filter(Boolean);
    return new Set(words);
  }

  async loadAcronyms(): Promise<Record<string, string>> {
    const res = await fetch(this.baseUrl + "/acronym.csv");
    const text = await res.text();

    const dict: Record<string, string> = {};
    for (const line of text.split(/\r?\n/)) {
      const [acro, exp] = line.split(",");
      if (acro && exp) dict[acro.trim()] = exp.trim();
    }
    return dict;
  }

  // --------------------------------------------------
  // Vocab and Idf
  // --------------------------------------------------

  async loadVocab(): Promise<Record<string, number>> {
    const res = await fetch(this.baseUrl + "/vocab.json");
    if (!res.ok) throw new Error(`Failed to load vocab.json: ${res.status}`);

    const vocab = await res.json();
    return vocab as Record<string, number>;
  }

  async loadIdf(): Promise<Float32Array> {
    const res = await fetch(this.baseUrl + "/idf.f32");
    if (!res.ok) throw new Error(`Failed to load idf.f32: ${res.status}`);

    const buf = await res.arrayBuffer();
    return new Float32Array(buf);
  }

  // --------------------------------------------------
  // MANIFEST
  // --------------------------------------------------

  async loadManifest(): Promise<void> {
    const res = await fetch(`${this.baseUrl}/manifest.json`);
    if (!res.ok) throw new Error("Failed to load manifest.json");
    this.manifest = await res.json();
  }

  getManifest(): VectorManifest {
    if (!this.manifest) {
      throw new Error("Manifest not loaded. Call loadManifest()");
    }
    return this.manifest;
  }

  getShardInfo(id: number): EmbeddingShard {
    const manifest = this.getManifest();
    const info = manifest.shards[id];
    if (!info) throw new Error(`Shard ${id} not found in manifest`);
    return info;
  }

  // --------------------------------------------------
  // VECTOR SHARDS (.f32)
  // --------------------------------------------------

  async loadShard(id: number): Promise<Float32Array> {
    if (this.shardCache.has(id)) {
      return this.shardCache.get(id)!;
    }

    const shardInfo = this.getShardInfo(id);

    const url = `${this.baseUrl}/${shardInfo.shard}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch shard: ${url}`);

    const arrayBuffer = await res.arrayBuffer();
    const float32 = new Float32Array(arrayBuffer);

    this.shardCache.set(id, float32);
    return float32;
  }

  // Optional: unload a shard to save memory
  unloadShard(id: number): void {
    this.shardCache.delete(id);
    console.log("unload",this.shardCache)
  }

  // --------------------------------------------------
  // METADATA INDEX (metadata.index)
  // --------------------------------------------------

  async loadMetadataIndex(): Promise<void> {
    const url = `${this.baseUrl}/metadata.index`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load metadata.index");

    const text = await res.text();

    // metadata.index is a newline-separated list of byte offsets
    this.metadataOffsets = text
      .trim()
      .split("\n")
      .map((x) => Number(x.trim()));
  }

  getMetadataIndex(): number[] {
    if (!this.metadataOffsets) {
      throw new Error("metadata.index not loaded. Call loadMetadataIndex()");
    }
    return this.metadataOffsets;
  }

  // --------------------------------------------------
  // METADATA FETCHING (JSONL via range)
  // --------------------------------------------------

  /** Fetch one line of metadata JSON via HTTP Range request */
  async fetchMetadataLine(line: number): Promise<MetadataObject> {
    const offsets = this.getMetadataIndex();

    if (line < 0 || line >= offsets.length - 1) {
      throw new Error(`Line ${line} out of range`);
    }

    const start = offsets[line];
    const end = offsets[line + 1] - 1;

    const url = `${this.baseUrl}/metadata.jsonl`;

    const res = await fetch(url, {
      headers: {
        Range: `bytes=${start}-${end}`,
      },
    });

    if (!res.ok && res.status !== 206) {
      throw new Error(`Range request failed for metadata line ${line}`);
    }

    const text = await res.text();
    return JSON.parse(text) as MetadataObject;
  }

  /** Fetch multiple metadata lines (may send many small range requests) */
  async fetchMetadataBatch(lines: number[]): Promise<MetadataObject[]> {
    const results: MetadataObject[] = [];

    for (const line of lines) {
      const item = await this.fetchMetadataLine(line);
      results.push(item);
    }

    return results;
  }
}
