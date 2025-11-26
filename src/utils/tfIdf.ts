export class QueryVectorizer {
  private vocab: Record<string, number>;
  private idf: Float32Array;
  private acronymDict: Record<string, string>;
  private acronymRegex: RegExp;
  private stopwords: Set<string>;

  constructor(
    vocab: Record<string, number>,
    idf: Float32Array,
    acronymDict: Record<string, string>,
    stopwords: Set<string>
  ) {
    this.vocab = vocab;
    this.idf = idf;
    this.stopwords = stopwords;
    this.acronymDict = acronymDict;
    this.acronymRegex = buildAcronymRegex(acronymDict);
  }

  vectorize(rawText: string): Float32Array {
    const cleaned = preprocessText(
      rawText,
      this.acronymDict,
      this.acronymRegex,
      this.stopwords
    );
    const tokens = tokenize(cleaned);

    // TF
    const counts: Record<string, number> = {};
    for (const t of tokens) counts[t] = (counts[t] || 0) + 1;

    const total = tokens.length;
    for (const t in counts) counts[t] /= total;

    const dim = this.idf.length;
    const vec = new Float32Array(dim);

    for (const term in counts) {
      const index = this.vocab[term];
      if (index !== undefined) {
        vec[index] = counts[term] * this.idf[index];
      }
    }

    // L2 normalize
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);

    if (norm > 0) {
      for (let i = 0; i < dim; i++) vec[i] /= norm;
    }

    return vec;
  }
}

function buildAcronymRegex(dict: Record<string, string>): RegExp {
  const escaped = Object.keys(dict).map((k) =>
    k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  );
  return new RegExp(`\\b(${escaped.join("|")})\\b`, "gi");
}

function preprocessText(
  text: string,
  acronymDict: Record<string, string>,
  acronymRegex: RegExp,
  stopwords: Set<string>
): string {
  text = String(text);

  // 1. Replace acronyms
  text = text.replace(
    acronymRegex,
    (match) => acronymDict[match.toUpperCase()] || acronymDict[match] || match
  );

  // 2. Lowercase
  text = text.toLowerCase();

  // 3. Remove <img ...> HTML tags
  text = text.replace(/<img[^>]*>/gi, "");

  // 4. Remove mentions, URLs, numbers
  text = text.replace(/@\w+/g, "");
  text = text.replace(/http\S+/g, "");
  text = text.replace(/\d+/g, "");

  // 5. Clean punctuation + whitespace
  text = text.replace(/b'/g, "");
  text = text.replace(/-/g, " ");
  text = text.replace(/[^\w\s]/g, ""); // same as Python
  text = text.replace(/\s+/g, " ").trim();

  // 6. Remove unwanted terms
  text = text.replace(/\bimg\b/g, "").replace(/\bsrc\b/g, "");

  // 7. Remove stopwords (Python does this in TfidfVectorizer automatically)
  const tokens = text.split(" ").filter((t) => t && !stopwords.has(t));

  return tokens.join(" ");
}

function tokenize(preprocessed: string): string[] {
  return preprocessed.split(" ").filter(Boolean);
}
