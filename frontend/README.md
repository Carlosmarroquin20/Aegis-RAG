# Aegis-RAG — Web Console

A small React + TypeScript single-page app that demonstrates the Aegis-RAG API:
ask grounded questions, index documents, and watch the `SecurityGateway` block
prompt-injection payloads in real time.

## Stack

- **React 18** + **TypeScript** (strict) + **Vite 5**
- **Tailwind CSS 3**
- No state library — a couple of hooks and the `fetch` API are enough.

## Features

- **Query** — submit a question, see the grounded answer, the retrieved sources
  with relevance scores, and the threat level assigned by the gateway.
- **Documents** — upload a TXT / MD / PDF / DOCX file and index it.
- **Security demo** — fire curated adversarial probes at the live gateway and see
  which are blocked (with the threat level and rejection reason) before any
  retrieval or generation happens.

The `X-API-Key` is entered in the UI and kept only in the browser's
`localStorage`; it is never bundled into the build.

## Run it

### With the full stack (recommended)

```bash
docker compose up -d      # from the repo root
```

Then open the console at **http://localhost:8080**. nginx serves the built app and
reverse-proxies `/api` to the API, so the browser talks to a single origin.

Use the dev API key `dev-key-change-in-production` (from `docker-compose.yml`).

### Local dev server

```bash
cd frontend
npm install
npm run dev               # http://localhost:5173
```

The Vite dev server proxies `/api`, `/health` and `/ready` to
`http://localhost:8000`, so run the API (or the compose stack) alongside it.

### Configuration

`VITE_API_BASE_URL` (see `.env.example`) points the built app at a specific API
origin. Leave it empty for same-origin (the default for both the nginx image and
the dev proxy).

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Start the Vite dev server |
| `npm run build` | Type-check (`tsc --noEmit`) and produce the production bundle |
| `npm run preview` | Preview the production build locally |
| `npm run typecheck` | Type-check only |
