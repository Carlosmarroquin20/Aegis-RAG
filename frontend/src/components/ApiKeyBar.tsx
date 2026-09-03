import { useState } from "react";

export function ApiKeyBar({
  apiKey,
  onChange,
}: {
  apiKey: string;
  onChange: (value: string) => void;
}) {
  const [reveal, setReveal] = useState(false);

  return (
    <div className="card flex flex-wrap items-center gap-3 px-4 py-3">
      <label htmlFor="api-key" className="text-sm font-medium text-slate-300">
        API key
      </label>
      <input
        id="api-key"
        className="field max-w-md flex-1 font-mono"
        type={reveal ? "text" : "password"}
        placeholder="X-API-Key (e.g. dev-key-change-in-production)"
        value={apiKey}
        autoComplete="off"
        spellCheck={false}
        onChange={(e) => onChange(e.target.value)}
      />
      <button type="button" className="btn-ghost" onClick={() => setReveal((r) => !r)}>
        {reveal ? "Hide" : "Show"}
      </button>
      <span className="text-xs text-slate-500">
        Sent as <code className="text-slate-400">X-API-Key</code>. Stored only in your browser.
      </span>
    </div>
  );
}
