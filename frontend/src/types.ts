// Mirrors the Aegis-RAG API DTOs (see src/aegis/application/dtos).

export type ThreatLevel = "CLEAN" | "SUSPICIOUS" | "BLOCKED";

export interface SourceDocument {
  content_preview: string;
  metadata: Record<string, string>;
  relevance_score: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  query_hash: string;
  threat_level: ThreatLevel;
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
