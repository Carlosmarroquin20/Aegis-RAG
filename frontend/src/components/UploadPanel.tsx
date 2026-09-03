import { useRef, useState } from "react";

import { uploadDocument } from "../api";
import { ApiError, type IngestResponse } from "../types";

export function UploadPanel({ apiKey }: { apiKey: string }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (!file || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      setResult(await uploadDocument(apiKey, file));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed. Is the API up?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={run} className="card space-y-4 p-5">
      <div>
        <h2 className="text-sm font-semibold text-slate-200">Index a document</h2>
        <p className="mt-1 text-xs text-slate-500">
          TXT, Markdown, PDF or DOCX. The MIME type is detected from magic bytes, not the file
          extension. Max 50 MB.
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        className="block w-full text-sm text-slate-400 file:mr-3 file:rounded-lg file:border-0 file:bg-aegis-accent/15 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-aegis-accent hover:file:bg-aegis-accent/25"
      />

      <button type="submit" className="btn-primary" disabled={loading || !file}>
        {loading ? "Indexing…" : "Upload & index"}
      </button>

      {error && <p className="text-sm text-amber-300">{error}</p>}

      {result && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-4 text-sm">
          <p className="font-medium text-emerald-300">Indexed “{result.source}”</p>
          <ul className="mt-2 space-y-1 text-slate-300">
            <li>
              Chunks created: <span className="font-mono">{result.chunks_created}</span>
            </li>
            <li>
              Duplicates skipped: <span className="font-mono">{result.duplicates_skipped}</span>
            </li>
            <li>
              Collection: <span className="font-mono">{result.collection}</span>
            </li>
          </ul>
          {result.warnings.length > 0 && (
            <ul className="mt-2 list-inside list-disc text-xs text-amber-300">
              {result.warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </form>
  );
}
