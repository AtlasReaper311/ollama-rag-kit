<div align="center">
  <img src="https://raw.githubusercontent.com/AtlasReaper311/AtlasReaper311/main/atlas-icon-dark-256.png" width="88" alt="Atlas Systems"/>
</div>

# ollama-rag-kit

```
┌─────────────────────────────────────────────┐
│  ATLAS SYSTEMS // ollama-rag-kit            │
│  ask questions of your own documents,       │
│  entirely on your own hardware              │
└─────────────────────────────────────────────┘
```

![Python](https://img.shields.io/badge/python-3.12-f5a623?style=flat-square&labelColor=0a0a0f)
![FastAPI](https://img.shields.io/badge/api-fastapi-4ade80?style=flat-square&labelColor=0a0a0f)
![Vector store](https://img.shields.io/badge/vectors-chromadb-aaa9a0?style=flat-square&labelColor=0a0a0f)
![Cost](https://img.shields.io/badge/cost-%C2%A30-aaa9a0?style=flat-square&labelColor=0a0a0f)

A containerised retrieval-augmented generation pipeline for Ramone.
Generation and conversation memory stay local, while document retrieval
now comes from `atlas-corpus`, the estate-wide search index. The old
local `docs/` ingest path is retained for development experiments only;
production answers no longer search that nine-document subset.

```
POST /ask
   │
   ├──▶ atlas-corpus /search on SPECULAR-CORE ──▶ ranked context blocks
   │
   ├──▶ ChromaDB ramone_sessions                 ──▶ optional memory context
   │
   ▼
llama3.1:8b on Ollama ──▶ answer + cited sources
```

## Prerequisites

- Docker Desktop (Windows or macOS) or Docker Engine with the compose plugin (Linux/WSL2)
- [Ollama](https://ollama.com) running on the host, with two models:

  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b
  ```

Ollama stays on the host rather than in compose because it owns the GPU; the containers only need HTTP access to it.

## Setup

```bash
cp .env.example .env        # optional: defaults work out of the box
docker compose up --build -d
```

First boot waits for Ollama and Chroma, checks the configured models,
then serves on port 8000. It does not ingest `docs/`; atlas-corpus is
the ingest and retrieval source for production. Watch startup:

```bash
docker compose logs -f rag-api
```

Verify, then ask a question. Start atlas-corpus first or `/ask` will
fail clearly with a 503 instead of answering from an empty local index:

```bash
curl -sS http://localhost:8000/health
curl -sS -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What runs at api.atlas-systems.uk?"}'
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask `
  -ContentType "application/json" `
  -Body '{"question": "What runs at api.atlas-systems.uk?"}'
```

The answer cites its sources by block number, and the `sources` array carries file, chunk index, similarity score, and a text preview for each one.

## Usage

| Method and path | Purpose |
|---|---|
| `POST /ask` | `{"question": "...", "top_k": 4}` (top_k optional) |
| `POST /ask/stream` | Server-Sent Events for ramone-edge; same request shape |
| `GET /health` | Dependency-aware status; `503` when degraded |
| `GET /docs` | Interactive OpenAPI docs (FastAPI built-in) |

Tuning lives in `.env`; every knob and its default is listed in [`.env.example`](.env.example) and typed in [`app/config.py`](app/config.py).

## Retrieval source

Ramone retrieves from atlas-corpus over HTTP:

```env
ATLAS_CORPUS_URL=http://host.docker.internal:8092
```

`rag-api` runs in a different compose project from atlas-corpus, so
`localhost:8092` inside this container would point back at itself.
`host.docker.internal` uses the same host-gateway route already proven
by `OLLAMA_HOST`, stays on SPECULAR-CORE, avoids Cloudflare Tunnel
latency, and does not spend the public search rate budget. Override it
only if the deployment topology changes.

Every retrieval request sends `X-Atlas-Internal: ollama-rag-kit`.
atlas-corpus treats that header from private addresses as machine
traffic, exempting Ramone from visitor rate limits and analytics.

`docs/`, `app/ingest.py`, and the `atlas_docs` Chroma collection are
kept for local development and data-retention safety. They are no
longer read by production `/ask` or `/ask/stream`, and `/ingest/refresh`
has been removed. Do not delete the stored `atlas_docs` vectors without
an explicit cleanup decision, because Chroma also backs
`ramone_sessions` memory. See `docs/README-corpus-retrieval.md` for the
full migration notes.

Quick verification after changes:

1. Call `/ask/stream` with `why Cloudflare Workers over a VPS` and
   confirm the first event is `sources` with ids from the wider corpus,
   not old local `docs/` filenames.
2. Stop atlas-corpus temporarily and confirm `/ask` returns 503 in a
   few seconds, while `/ask/stream` emits
   `retrieval_failed:CorpusUnreachable` followed by `done`.
3. Run `pytest tests/test_retriever_corpus.py -q`.

## Networking: how containers reach host Ollama

The containers call `http://host.docker.internal:11434`. What makes that name resolve differs by platform:

| Where Docker runs | Where Ollama runs | Works because |
|---|---|---|
| Docker Desktop (Windows) | Windows host | Docker Desktop provides `host.docker.internal` natively |
| Docker Desktop (WSL2 backend) | Windows host | Same; Desktop routes to the Windows host |
| Docker Engine in WSL2 | WSL2 | `extra_hosts: host-gateway` in the compose file maps the name |
| Docker Engine on Linux | Same machine | Same `host-gateway` mapping |

**The Windows gotcha.** Ollama on Windows binds to `127.0.0.1` by default, which containers cannot reach even when the hostname resolves. Fix it once on the host:

```powershell
setx OLLAMA_HOST 0.0.0.0
```

then quit and restart Ollama from the system tray. The symptom of skipping this is the API logging `Waiting for Ollama (attempt n/30)` thirty times and exiting with a pointer to this section.

> **Running Ollama inside WSL2 instead of on Windows?** Then `host.docker.internal` points at the wrong place (the Windows host, not the WSL2 distro). Bind Ollama to `0.0.0.0` with a systemd override and set `OLLAMA_HOST` in `.env` to the distro's IP. That IP can change on a Windows reboot, so if generation starts failing after a restart, re-check it and `docker compose restart rag-api`.

## How it fits into Atlas Systems

This kit is the deployable sibling of [`atlas-kit-python-rag`](https://github.com/AtlasReaper311/atlas-kit-python-rag): that repo proves the pipeline as a tested, CI-green Python library; this one proves the deployed Ramone service shape. Retrieval is now delegated to [`atlas-corpus`](https://github.com/AtlasReaper311/atlas-corpus); this repo owns local memory, prompt assembly, generation, auth, and the streaming contract that `ramone-edge` forwards.

Optional event-bus hookup: set `NOTIFY_URL` and `NOTIFY_TOKEN` in `.env` and the service reports startup failures to [`atlas-notify`](https://github.com/AtlasReaper311/atlas-notify). Both empty by default; the kit stands alone.

---

Part of [atlas-systems.uk](https://atlas-systems.uk)
