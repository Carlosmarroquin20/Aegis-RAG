import type { ThreatLevel } from "../types";

const STYLES: Record<ThreatLevel, { label: string; className: string; dot: string }> = {
  CLEAN: {
    label: "Clean",
    className: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
    dot: "bg-emerald-400",
  },
  SUSPICIOUS: {
    label: "Suspicious",
    className: "border-amber-500/40 bg-amber-500/10 text-amber-300",
    dot: "bg-amber-400",
  },
  BLOCKED: {
    label: "Blocked",
    className: "border-red-500/40 bg-red-500/10 text-red-300",
    dot: "bg-red-400",
  },
};

export function ThreatBadge({ level }: { level: ThreatLevel }) {
  const style = STYLES[level] ?? STYLES.CLEAN;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${style.className}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
      {style.label}
    </span>
  );
}
