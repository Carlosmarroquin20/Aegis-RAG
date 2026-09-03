import { ApiError, type IngestResponse, type QueryResponse, type SecurityRejection } from "./types";

// Empty base ("") means same-origin: in dev the Vite proxy forwards /api to the
// backend; in the nginx image, nginx proxies it. Override with VITE_API_BASE_URL
// to point the built app at a remote API.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

const API_KEY_HEADER = "X-API-Key";

function authHeaders(apiKey: string): Record<string, string> {
  return apiKey ? { [API_KEY_HEADER]: apiKey } : {};
}

async function parseError(response: Response): Promise<never> {
  let detail: unknown;
  try {
    detail = (await response.json())?.detail;
  } catch {
    detail = undefined;
  }

  // A blocked query returns detail as a structured object.
  if (detail && typeof detail === "object" && "message" in detail) {
    const rejection = detail as SecurityRejection;
    throw new ApiError(response.status, rejection.message, rejection);
  }

  const message = typeof detail === "string" ? detail : `Request failed (${response.status}).`;
  throw new ApiError(response.status, message);
}

export async function submitQuery(
  apiKey: string,
  query: string,
  topK: number,
): Promise<QueryResponse> {
  const response = await fetch(`${BASE_URL}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(apiKey) },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!response.ok) await parseError(response);
  return response.json();
}

export async function uploadDocument(
  apiKey: string,
  file: File,
  opts: { chunkSize?: number; overlap?: number } = {},
): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  if (opts.chunkSize != null) form.append("chunk_size", String(opts.chunkSize));
  if (opts.overlap != null) form.append("overlap", String(opts.overlap));

  const response = await fetch(`${BASE_URL}/api/v1/documents`, {
    method: "POST",
    headers: authHeaders(apiKey),
    body: form,
  });
  if (!response.ok) await parseError(response);
  return response.json();
}

export async function checkReady(): Promise<{ status: string; components: Record<string, boolean> }> {
  const response = await fetch(`${BASE_URL}/ready`);
  return response.json();
}
