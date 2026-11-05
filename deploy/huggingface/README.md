---
title: Aegis-RAG
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Security-first RAG. Try to get past the gateway.
---

# Aegis-RAG — public demo

A hardened Retrieval-Augmented Generation API: every question is scored by a
security gateway (OWASP LLM01) before retrieval, and every answer is checked by
an output sanitizer (LLM05) before it is returned.

- **Ask** about the policies of *Tienda Modelo, S.A.*, a fictional retailer
  (Spanish or English).
- **Attack** it from the *Security demo* tab: instruction overrides, persona
  hijacks, prompt-leak probes, in English and Spanish.

This instance uses a hosted LLM, so questions leave the server. Do not type
anything sensitive. The production design runs air-gapped on a local model.
The corpus is read-only and all visitors share one rate limit and a daily token
budget.

Source, architecture and threat model:
<https://github.com/Carlosmarroquin20/Aegis-RAG>

This Space is generated from `deploy/huggingface/` in that repository; edit it
there, not here.
