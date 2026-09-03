import type { SourceDocument } from "../types";

export function SourceList({ sources }: { sources: SourceDocument[] }) {
  if (sources.length === 0) {
    return <p className="text-sm text-slate-500">No source documents were retrieved.</p>;
  }

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Sources ({sources.length})
      </h3>
      <ul className="space-y-2">
        {sources.map((source, index) => (
          <li key={index} className="rounded-lg border border-aegis-border bg-aegis-bg/40 p-3">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="truncate font-mono text-xs text-slate-400">
                {source.metadata.source ?? "unknown"}
              </span>
              <span className="shrink-0 rounded bg-aegis-accent/15 px-1.5 py-0.5 font-mono text-[11px] text-aegis-accent">
                {source.relevance_score.toFixed(3)}
              </span>
            </div>
            <p className="text-sm text-slate-300">{source.content_preview}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
