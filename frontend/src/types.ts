// Mirrors the Aegis-RAG API DTOs (see src/aegis/application/dtos).

export type ThreatLevel = "CLEAN" | "SUSPICIOUS" | "BLOCKED";

export interface SourceDocument {
  content_preview: string;
  metadata: Record<string, string>;
  relevance_score: number;
}

export interface FaithfulnessReport {
  score: number;
  abstained: boolean;
  unsupported: string[];
}

export interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  query_hash: string;
  threat_level: ThreatLevel;
  faithfulness?: FaithfulnessReport | null;
}

// Events of POST /api/v1/query/stream (server-sent events).
export interface StreamStart {
  query_hash: string;
  threat_level: ThreatLevel;
  sources: SourceDocument[];
}

export interface StreamEnd {
  answer: string;
  truncated: boolean;
  faithfulness?: FaithfulnessReport | null;
}

export interface StreamHandlers {
  onStart: (start: StreamStart) => void;
  onDelta: (text: string) => void;
  onEnd: (end: StreamEnd) => void;
}

export interface IngestResponse {
  source: string;
  chunks_created: number;
  duplicates_skipped: number;
  warnings: string[];
  collection: string;
}

// Structured 400 returned when the SecurityGateway blocks a query. The route
// wraps this under FastAPI's `detail` envelope.
export interface SecurityRejection {
  message: string;
  query_hash: string;
  threat_level: ThreatLevel;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public rejection?: SecurityRejection,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
