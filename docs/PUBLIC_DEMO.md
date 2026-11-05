# Public demo

A hosted copy of the web console that anyone can try from a link, on free hosting, with no
clone and no Docker. It runs on [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
(free CPU tier) with [Groq](https://console.groq.com)'s free tier as the LLM. Neither asks for
a payment card, and neither can bill: running out of free quota means `429`, not a charge.

The reasoning behind its shape is in [ADR 0024](adr/0024-public-demo-on-free-hosting.md).

## What a visitor gets

- The console with the demo key already set, a notice that questions go to a hosted LLM,
  and example questions.
- **Query**: answers from the seven policy documents of a fictional retailer
  ([`examples/atencion-cliente`](../examples/atencion-cliente)), in Spanish or English.
- **Security demo**: the curated injection probes, against the real gateway.
- **Documents**: the list of indexed files. Uploads and deletes are disabled.

## How it differs from the compose stack

| | Compose stack | Public demo |
|---|---|---|
| Containers | 8 (API, Chroma, Ollama, Redis, Prometheus, Grafana, OTel Collector, Jaeger) + console | 1: nginx + API + Chroma in-process |
| LLM | Ollama, local (air-gapped) | Groq, hosted: questions leave the server |
| Corpus | Whatever you upload | Indexed at build time, read-only |
| API keys | Yours, secret | One, public, baked into the console |
| Abuse limits | Per tenant | Shared by every visitor: 20 requests/min, 400k tokens/day |
| Observability | Prometheus, Grafana, Jaeger | Logs only; `/metrics` is not exposed |

## Deploying it

One-time setup, about ten minutes:

1. **Groq key.** Create a free account at <https://console.groq.com> and an API key.
2. **Space.** On Hugging Face, *New Space* → SDK **Docker** → blank template → hardware
   **CPU basic (free)** → visibility **Public**. Note its id, `<your-user>/<space-name>`.
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

### Without GitHub Actions

```bash
deploy/huggingface/stage.sh build/hf-space
```

then push the contents of `build/hf-space` to the Space's git repository
(`https://huggingface.co/spaces/<your-user>/<space-name>`).

### Running the image locally

```bash
docker build -f deploy/huggingface/Dockerfile -t aegis-rag-demo .
docker run --rm -p 7860:7860 -e LLM_API_KEY=<groq-key> aegis-rag-demo
```

and open <http://localhost:7860>.

## Tuning

Every setting can be overridden with a Space *variable* (not a secret), without a rebuild:

- `LLM_MODEL` (default `openai/gpt-oss-20b`) and `LLM_BASE_URL` (default Groq). Any
  OpenAI-compatible provider works. Providers retire models: if every question answers `503`
  and the log says `model_not_found`, pick a current one from the provider's model list.
- The request limit and the daily token budget are in
  [`deploy/huggingface/tenants.toml`](../deploy/huggingface/tenants.toml). Keep the budget below
  the provider's free daily allowance for the chosen model (Groq lists them under *Limits*);
  changing it needs a redeploy.

## Limits worth knowing

- The free Space sleeps after about two days without visits. The next visitor waits for the
  container to start (around a minute: the embedding model is loaded from the image).
- All visitors share one rate limit and one daily budget, so one determined caller can make the
  demo answer `429` for everyone until the minute, or the UTC day, turns over. That is the
  intended worst case: it costs nothing.
- The key is public by design. It authorizes asking questions, within those limits, and
  nothing else.
