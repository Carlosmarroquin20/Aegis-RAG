import { useState } from "react";

import { submitQuery } from "../api";
import { ApiError, type QueryResponse } from "../types";
import { SourceList } from "./SourceList";
import { ThreatBadge } from "./ThreatBadge";

export function QueryPanel({ apiKey }: { apiKey: string }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      setResult(await submitQuery(apiKey, query, topK));
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "Network error. Is the API up?"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <form onSubmit={run} className="card space-y-4 p-5">
        <textarea
          className="field min-h-[120px] resize-y"
          placeholder="Ask a question about your indexed documents…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-400">
            top_k
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="field w-20"
            />
          </label>
          <button type="submit" className="btn-primary" disabled={loading || !query.trim()}>
            {loading ? "Evaluating…" : "Ask Aegis"}
          </button>
        </div>
        <p className="text-xs text-slate-500">
          Every query is scored by the SecurityGateway before retrieval, and the answer is
          post-processed by the OutputSanitizer before it reaches you.
        </p>
      </form>

      {error && <ErrorPanel error={error} />}

      {result && (
        <div className="card space-y-4 p-5">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Answer</h2>
            <ThreatBadge level={result.threat_level} />
          </div>
          <p className="whitespace-pre-wrap text-slate-100">{result.answer}</p>
          <SourceList sources={result.sources} />
          <p className="font-mono text-[11px] text-slate-600">query_hash: {result.query_hash}</p>
        </div>
      )}
    </div>
  );
}

function ErrorPanel({ error }: { error: ApiError }) {
  if (error.rejection) {
    return (
      <div className="card space-y-2 border-red-500/40 p-5">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-red-300">
            Blocked by the SecurityGateway
          </h2>
          <ThreatBadge level={error.rejection.threat_level} />
        </div>
        <p className="text-slate-300">{error.rejection.message}</p>
        <p className="font-mono text-[11px] text-slate-600">
          query_hash: {error.rejection.query_hash}
        </p>
      </div>
    );
  }

  const hint =
    error.status === 403
      ? "Missing or invalid API key — set it above."
      : error.status === 429
        ? "Rate limit exceeded — slow down and retry."
        : "Check that the API is running and reachable.";

  return (
    <div className="card border-amber-500/40 p-5">
      <h2 className="text-sm font-semibold text-amber-300">
        Request failed{error.status ? ` (${error.status})` : ""}
      </h2>
      <p className="mt-1 text-slate-300">{error.message}</p>
      <p className="mt-1 text-xs text-slate-500">{hint}</p>
    </div>
  );
}
