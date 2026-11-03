import {
  ApiError,
  type IngestResponse,
  type QueryResponse,
  type SecurityRejection,
  type StreamHandlers,
} from "./types";

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

/**
 * Streams an answer from POST /api/v1/query/stream.
 *
 * Refusals (blocked query, rate limit, unavailable backend) arrive as ordinary
 * HTTP errors before any event and are thrown as ApiError. An `error` event
 * means the answer failed after text was shown: it is thrown too, and the caller
 * must discard the partial answer. The `end` event's answer is authoritative and
 * replaces the text assembled from the deltas.
 */
export async function streamQuery(
  apiKey: string,
  query: string,
  topK: number,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE_URL}/api/v1/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...authHeaders(apiKey),
    },
    body: JSON.stringify({ query, top_k: topK }),
    signal,
  });
  if (!response.ok) await parseError(response);
  if (!response.body) throw new ApiError(0, "The browser did not expose the response stream.");

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let ended = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // Events are separated by a blank line.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      ended = dispatch(block, handlers) || ended;
      boundary = buffer.indexOf("\n\n");
    }
  }
  if (!ended) throw new ApiError(0, "The answer stream ended early. Discard what was shown.");
}

/** Handles one event block; returns true when it ended the stream. */
function dispatch(block: string, handlers: StreamHandlers): boolean {
  let name = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return false;
  const payload = JSON.parse(data.join("\n"));
  switch (name) {
    case "start":
      handlers.onStart(payload);
      return false;
    case "delta":
      handlers.onDelta(payload.text);
      return false;
    case "end":
      handlers.onEnd(payload);
      return true;
    case "error":
      throw new ApiError(500, payload.message);
    default:
      return false;
  }
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
