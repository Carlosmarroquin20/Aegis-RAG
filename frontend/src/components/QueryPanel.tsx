import { useRef, useState } from "react";

import { streamQuery } from "../api";
import { ApiError, type FaithfulnessReport, type StreamStart } from "../types";
import { SourceList } from "./SourceList";
import { ThreatBadge } from "./ThreatBadge";

export function QueryPanel({ apiKey }: { apiKey: string }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [start, setStart] = useState<StreamStart | null>(null);
  const [answer, setAnswer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [faithfulness, setFaithfulness] = useState<FaithfulnessReport | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const abort = useRef<AbortController | null>(null);

  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim() || loading) return;
    abort.current = new AbortController();
    setLoading(true);
    setStart(null);
    setAnswer("");
    setFaithfulness(null);
    setError(null);
    try {
      await streamQuery(
        apiKey,
        query,
        topK,
        {
          onStart: (s) => {
            setStart(s);
            setStreaming(true);
          },
          onDelta: (text) => setAnswer((shown) => shown + text),
          onEnd: (end) => {
            // The final answer is authoritative: it replaces the assembled deltas.
            setAnswer(end.answer);
            setFaithfulness(end.faithfulness ?? null);
          },
        },
        abort.current.signal,
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      // A failure after text was shown means that text must not be trusted.
      setStart(null);
      setAnswer("");
      setError(err instanceof ApiError ? err : new ApiError(0, "Network error. Is the API up?"));
    } finally {
      setStreaming(false);
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
          <div className="flex items-center gap-2">
            {loading && (
              <button type="button" className="btn-ghost" onClick={() => abort.current?.abort()}>
                Stop
              </button>
            )}
            <button type="submit" className="btn-primary" disabled={loading || !query.trim()}>
              {loading ? (streaming ? "Answering…" : "Evaluating…") : "Ask Aegis"}
            </button>
          </div>
        </div>
        <p className="text-xs text-slate-500">
          Every query is scored by the SecurityGateway before retrieval, and the answer streams
          in a sentence at a time, each one checked by the OutputSanitizer before it reaches you.
        </p>
      </form>

      {error && <ErrorPanel error={error} />}

      {start && (
        <div className="card space-y-4 p-5">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Answer</h2>
            <ThreatBadge level={start.threat_level} />
          </div>
          <p className="whitespace-pre-wrap text-slate-100">
            {answer}
            {streaming && <span className="ml-0.5 animate-pulse text-slate-500">▍</span>}
          </p>
          {faithfulness && <FaithfulnessNote report={faithfulness} />}
          <SourceList sources={start.sources} />
          <p className="font-mono text-[11px] text-slate-600">query_hash: {start.query_hash}</p>
        </div>
      )}
    </div>
  );
}

function FaithfulnessNote({ report }: { report: FaithfulnessReport }) {
  if (report.abstained || report.unsupported.length === 0) return null;
  return (
    <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-sm">
      <p className="text-amber-300">
        {report.unsupported.length === 1
          ? "One sentence is not supported by the retrieved sources:"
          : `${report.unsupported.length} sentences are not supported by the retrieved sources:`}
      </p>
      <ul className="mt-1 list-disc pl-5 text-slate-300">
        {report.unsupported.map((sentence) => (
          <li key={sentence}>{sentence}</li>
        ))}
      </ul>
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
