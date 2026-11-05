import { REPOSITORY_URL } from "../demo";

export function DemoNotice() {
  return (
    <div className="card space-y-1.5 border-aegis-accent/40 px-4 py-3 text-sm text-slate-300">
      <p>
        <span className="font-semibold text-white">Public demo.</span> Ask about the policies of a
        fictional retailer (in Spanish or English), or try to get past the gateway in the{" "}
        <span className="text-slate-100">Security demo</span> tab.
      </p>
      <p className="text-xs text-slate-500">
        This instance uses a hosted LLM, so questions leave the server: do not type anything
        sensitive. The production design runs air-gapped on a local model. Shared limits apply
        to every visitor. Source and architecture:{" "}
        <a className="text-aegis-accent underline" href={REPOSITORY_URL} rel="noreferrer">
          GitHub
        </a>
        .
      </p>
    </div>
  );
}
