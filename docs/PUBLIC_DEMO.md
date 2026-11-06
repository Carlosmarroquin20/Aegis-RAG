# Public demo

There are two ways to try Aegis-RAG without cloning it. Why there are two is in
[ADR 0024](adr/0024-public-demo-on-free-hosting.md).

| | Gateway playground | Full single-container demo |
|---|---|---|
| What runs | The SecurityGateway, in your browser (Pyodide) | Console + API + retrieval + LLM + sanitizer |
| Where | GitHub Pages: <https://carlosmarroquin20.github.io/Aegis-RAG/playground/> | Any Docker host (not published by default) |
| Cost | Nothing, for anyone | The host's price: Hugging Face Docker Spaces need a paid plan |
| Data | Never leaves the browser tab | Questions go to a hosted LLM |

## Gateway playground

A static page that loads [Pyodide](https://pyodide.org) and runs the gateway module from this
repository. Pick an attack or a legitimate question, or type your own, and see the verdict,
the score, the rules that fired and how long it took. There is no API, no key and no quota.

- **Source:** [`deploy/gateway-playground/`](../deploy/gateway-playground). `build.py` follows
  the gateway's imports from `src/` and copies exactly those files; the page loads them into
  Pyodide. `--verify` imports the built tree on its own and checks it blocks a known attack, so
  a missing file fails the build rather than a visitor's browser.
- **No third-party code at runtime:** the Pyodide runtime is served by the site itself. The
  build takes it from the `pyodide` npm package, checked against the SHA-512 integrity npm
  publishes (pinned in `build.py`), and the pydantic wheels against the SHA-256 in that
  runtime's lock file. The page's CSP allows `'self'` only, plus `'wasm-unsafe-eval'` to compile
  the runtime. GitHub Pages cannot send headers, so the policy is a `<meta>` tag, and framing
  cannot be restricted; the page has no state or action worth framing.
- **Publishing:** the *Publish gateway playground* workflow rebuilds and verifies it, on Python
  3.12 with the pydantic version Pyodide ships, and pushes it to `playground/` on the
  `gh-pages` branch whenever anything under `src/aegis/` changes on main (or by hand).
- **Locally** (downloads the runtime, about 15 MB; `--verify` needs pydantic 2.10.5):

  ```bash
  uv run --no-project --python 3.12 --with "pydantic==2.10.5" python deploy/gateway-playground/build.py --out build/playground --verify
  python -m http.server 8765 --directory build/playground
  ```

  and open <http://localhost:8765>.

## Full single-container demo

A single image (`deploy/huggingface/`) with nginx, the API and an in-process Chroma, the
example corpus indexed at build time, and a hosted LLM ([Groq](https://console.groq.com)'s
free tier by default). It runs on any Docker host; Hugging Face Spaces is the one with a
ready-made workflow, but its Docker Spaces need a paid plan.

### What a visitor gets

- The console with the demo key already set, a notice that questions go to a hosted LLM,
  and example questions.
- **Query**: answers from the seven policy documents of a fictional retailer
  ([`examples/atencion-cliente`](../examples/atencion-cliente)), in Spanish or English.
- **Security demo**: the curated injection probes, against the real gateway.
- **Documents**: the list of indexed files. Uploads and deletes are disabled.

### How it differs from the compose stack

| | Compose stack | Public demo |
|---|---|---|
| Containers | 8 (API, Chroma, Ollama, Redis, Prometheus, Grafana, OTel Collector, Jaeger) + console | 1: nginx + API + Chroma in-process |
| LLM | Ollama, local (air-gapped) | Groq, hosted: questions leave the server |
| Corpus | Whatever you upload | Indexed at build time, read-only |
| API keys | Yours, secret | One, public, baked into the console |
| Abuse limits | Per tenant | Shared by every visitor: 20 requests/min, 400k tokens/day |
| Observability | Prometheus, Grafana, Jaeger | Logs only; `/metrics` is not exposed |

### Deploying it to a Hugging Face Space

One-time setup, about ten minutes:

1. **Groq key.** Create a free account at <https://console.groq.com> and an API key.
2. **Space.** On Hugging Face, *New Space* → SDK **Docker** (needs a paid plan) → blank
   template → hardware **CPU basic** → visibility **Public**. Note its id, `<your-user>/<space-name>`.
3. **Space secret.** In the Space's *Settings → Variables and secrets*, add the secret
   `LLM_API_KEY` with the Groq key. It never leaves the Space; the image has no default and the
   API refuses to start without it.
4. **Deploy token.** In your Hugging Face settings, *Access Tokens* → a fine-grained token with
   write access to that Space only.
5. **GitHub.** In this repository's *Settings → Secrets and variables → Actions*, add the
   secret `HF_TOKEN` (the token) and the variable `HF_SPACE` (the Space id).

Then each deployment is: *Actions → Deploy public demo → Run workflow*. The workflow uploads
the image's sources to the Space ([`deploy/huggingface/stage.sh`](../deploy/huggingface/stage.sh)
assembles them from the committed tree), and the Space builds and starts the image. The first
build takes about ten minutes; the Space shows its build log.

#### Without GitHub Actions

```bash
deploy/huggingface/stage.sh build/hf-space
```

then push the contents of `build/hf-space` to the Space's git repository
(`https://huggingface.co/spaces/<your-user>/<space-name>`).

### Running the image locally or on another host

```bash
docker build -f deploy/huggingface/Dockerfile -t aegis-rag-demo .
docker run --rm -p 7860:7860 -e LLM_API_KEY=<groq-key> aegis-rag-demo
```

and open <http://localhost:7860>. Any host that runs a container with 2 GB of memory and
passes `LLM_API_KEY` as a secret will do.

### Tuning

Every setting can be overridden with an environment variable (a Space *variable*, not a
secret), without a rebuild:

- `LLM_MODEL` (default `openai/gpt-oss-20b`) and `LLM_BASE_URL` (default Groq). Any
  OpenAI-compatible provider works. Providers retire models: if every question answers `503`
  and the log says `model_not_found`, pick a current one from the provider's model list.
- The request limit and the daily token budget are in
  [`deploy/huggingface/tenants.toml`](../deploy/huggingface/tenants.toml). Keep the budget below
  the provider's free daily allowance for the chosen model (Groq lists them under *Limits*);
  changing it needs a redeploy.

### Limits worth knowing

- Hosts that scale to zero make the next visitor wait for the container to start (around a
  minute: the embedding model is loaded from the image).
- All visitors share one rate limit and one daily budget, so one determined caller can make the
  demo answer `429` for everyone until the minute, or the UTC day, turns over. That is the
  intended worst case: it costs nothing at the LLM provider.
- The key is public by design. It authorizes asking questions, within those limits, and
  nothing else.
