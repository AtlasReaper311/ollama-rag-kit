# Atlas Systems: stack overview

Atlas Systems is the infrastructure and portfolio operation behind
atlas-systems.uk, run by Atlas Reaper. It spans a public website, a set
of Cloudflare Workers on the api subdomain, and self-hosted AI
infrastructure on the primary workstation, SPECULAR-CORE.

## Services on api.atlas-systems.uk

Two Cloudflare Workers run on the api subdomain. atlas-notify lives at
api.atlas-systems.uk/notify and acts as the central event router: every
service in the stack posts events to it, and it forwards them to Discord
as colour-coded embeds. github-pulse lives at api.atlas-systems.uk/pulse
and serves cached GitHub activity statistics to the portfolio site,
keeping the GitHub token server-side and the rate limit unbothered.

## Local AI infrastructure

SPECULAR-CORE runs Ollama with several local models, including
qwen2.5:32b for general reasoning, deepseek-coder-v2:16b for code, and
nomic-embed-text for embeddings. Ramone is the umbrella name for this
self-hosted AI system: Ollama, Open WebUI, Docker, and a set of
specialised workbots.

ollama-rag-kit, the service answering this question right now, is part
of that infrastructure. It ingests documents from a local folder, embeds
them with nomic-embed-text, stores vectors in ChromaDB, and answers
questions grounded in those documents using a local LLM. Nothing leaves
the machine.

## Try asking

A good first question for this kit:

> What runs at api.atlas-systems.uk?

The answer should come back citing this document as its source. Replace
this file with your own Markdown, text, or PDF documents and re-run
ingest to make the kit yours.
