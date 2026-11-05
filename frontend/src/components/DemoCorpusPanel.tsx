import { DEMO_CORPUS } from "../demo";

export function DemoCorpusPanel() {
  return (
    <div className="card space-y-4 p-5">
      <div>
        <h2 className="text-sm font-semibold text-slate-200">Indexed documents</h2>
        <p className="mt-1 text-xs text-slate-500">
          Uploads and deletes are disabled on the public demo: ingested documents are trusted
          source material and do not pass through the gateway, so an open upload would let
          anyone plant text in other visitors' answers. The API answers those requests with 403.
        </p>
      </div>
      <ul className="space-y-1 font-mono text-sm text-slate-300">
        {DEMO_CORPUS.map((name) => (
          <li key={name}>{name}</li>
        ))}
      </ul>
      <p className="text-xs text-slate-500">
        Seven policy documents for <em>Tienda Modelo, S.A.</em>, a fictional retailer. Every
        figure in them is invented.
      </p>
    </div>
  );
}
