# Diagrams (diagram-as-code)

These diagrams are authored as small, validated **JSON specifications** and rendered to
self-contained, interactive HTML with [Archify](https://github.com/tt-a1i/archify). The spec is the
source of truth — the HTML is a build artifact — so the diagrams stay reviewable in pull requests and
never drift into stale screenshots.

| Spec | Rendered | Shows |
|---|---|---|
| [`aegis-architecture.json`](aegis-architecture.json) | [`aegis-architecture.html`](https://raw.githack.com/Carlosmarroquin20/Aegis-RAG/main/docs/diagrams/aegis-architecture.html) | Runtime topology, trust boundaries, observability fan-out |
| [`aegis-query-sequence.json`](aegis-query-sequence.json) | [`aegis-query-sequence.html`](https://raw.githack.com/Carlosmarroquin20/Aegis-RAG/main/docs/diagrams/aegis-query-sequence.html) | The security-first query pipeline, message by message |

GitHub does not render HTML inline, so the links above open the interactive version through
raw.githack; the raw `.html` files can also be opened locally in any browser.

## Regenerate

```bash
npx skills add tt-a1i/archify -g            # or clone https://github.com/tt-a1i/archify
node bin/archify.mjs validate architecture aegis-architecture.json --quality showcase --json
node bin/archify.mjs deliver  architecture aegis-architecture.json aegis-architecture.html --quality showcase --json
```

Each render must pass all nine showcase composition checks (single SVG, orthogonal arrows,
label/route clearance, no crossings, corridor spacing, route rhythm, legend clearance) before it
replaces the committed HTML. The interactive viewer also exports PNG/SVG from its **Export** menu.
