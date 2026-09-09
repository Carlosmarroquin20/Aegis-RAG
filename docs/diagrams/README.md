# Diagrams (diagram-as-code)

These diagrams are authored as small, validated **JSON specifications** and rendered to
self-contained, interactive HTML with [Archify](https://github.com/tt-a1i/archify). The spec is the
committed source of truth; the rendered `.html` is a build artifact and is **git-ignored** (it embeds
a font and a viewer runtime — ~15k lines each — which does not belong in the repo). This keeps the
diagrams reviewable in pull requests and never lets them drift into stale screenshots.

| Spec | Shows |
|---|---|
| [`aegis-architecture.json`](aegis-architecture.json) | Runtime topology, trust boundaries, observability fan-out |
| [`aegis-query-sequence.json`](aegis-query-sequence.json) | The security-first query pipeline, message by message |

Static previews live next to each spec (`*.png`) and are embedded in the top-level
[`README.md`](../../README.md#architecture).

## Render or export

```bash
npx skills add tt-a1i/archify -g            # or clone https://github.com/tt-a1i/archify
node bin/archify.mjs validate architecture aegis-architecture.json --quality showcase --json
node bin/archify.mjs deliver  architecture aegis-architecture.json aegis-architecture.html --quality showcase --json
```

Every render must pass Archify's nine showcase composition checks (single SVG, orthogonal arrows,
label/route clearance, no crossings, corridor spacing, route rhythm, legend clearance) before it is
accepted. Open the resulting `.html` in a browser to explore it (pan/zoom, guided views, theme
toggle) and use the viewer's **Export → PNG** to refresh the static preview.
