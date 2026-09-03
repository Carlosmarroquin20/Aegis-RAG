import { useEffect, useState } from "react";

import { checkReady } from "./api";
import { ApiKeyBar } from "./components/ApiKeyBar";
import { QueryPanel } from "./components/QueryPanel";
import { SecurityDemo } from "./components/SecurityDemo";
import { UploadPanel } from "./components/UploadPanel";
import { useLocalStorage } from "./hooks/useLocalStorage";

type Tab = "query" | "documents" | "security";

const TABS: { id: Tab; label: string }[] = [
  { id: "query", label: "Query" },
  { id: "documents", label: "Documents" },
  { id: "security", label: "Security demo" },
];

export default function App() {
  const [apiKey, setApiKey] = useLocalStorage("aegis.apiKey", "");
  const [tab, setTab] = useState<Tab>("query");

  return (
    <div className="mx-auto flex min-h-screen max-w-3xl flex-col gap-5 px-4 py-8">
      <Header />
      <ApiKeyBar apiKey={apiKey} onChange={setApiKey} />

      <nav className="flex gap-1 rounded-xl border border-aegis-border bg-aegis-panel/50 p-1">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            className={`tab flex-1 ${tab === id ? "tab-active" : ""}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      <main className="flex-1">
        {tab === "query" && <QueryPanel apiKey={apiKey} />}
        {tab === "documents" && <UploadPanel apiKey={apiKey} />}
        {tab === "security" && <SecurityDemo apiKey={apiKey} />}
      </main>

      <footer className="pt-2 text-center text-xs text-slate-600">
        Aegis-RAG — security is the architecture, not a checklist.
      </footer>
    </div>
  );
}

function Header() {
  return (
    <header className="flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <ShieldMark />
        <div>
          <h1 className="text-lg font-semibold text-white">Aegis-RAG Console</h1>
          <p className="text-xs text-slate-500">
            Hardened Retrieval-Augmented Generation · OWASP LLM Top 10
          </p>
        </div>
      </div>
      <ReadinessDot />
    </header>
  );
}

function ShieldMark() {
  return (
    <svg viewBox="0 0 24 24" className="h-8 w-8 text-aegis-accent" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 2 4 5v6c0 5 3.4 8.5 8 11 4.6-2.5 8-6 8-11V5l-8-3Zm0 2.2 6 2.25V11c0 3.9-2.5 6.8-6 8.8-3.5-2-6-4.9-6-8.8V6.45l6-2.25Z"
      />
      <path fill="currentColor" d="m11 13.4-2-2-1.4 1.4L11 16.2l5.4-5.4L15 9.4l-4 4Z" />
    </svg>
  );
}

function ReadinessDot() {
  const [ready, setReady] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    checkReady()
      .then((r) => active && setReady(r.status === "ready"))
      .catch(() => active && setReady(false));
    return () => {
      active = false;
    };
  }, []);

  const color =
    ready === null ? "bg-slate-500" : ready ? "bg-emerald-400" : "bg-red-400";
  const label = ready === null ? "checking" : ready ? "ready" : "unreachable";

  return (
    <span className="flex items-center gap-2 text-xs text-slate-400">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}
