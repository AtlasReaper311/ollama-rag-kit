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

A containerised retrieval-augmented generation pipeline. Drop Markdown,
text, or PDF files into `docs/`, run one command, and ask questions
over HTTP. Embeddings, vector storage, and generation all run locally;
no document, question, or answer leaves the machine.

```
docs/*.{md,txt,pdf}
   │  ingest on startup (idempotent, hash-keyed)
   ▼
nomic-embed-text ──▶ ChromaDB (cosine)        [containers]
                        │
POST /ask ──▶ embed ──▶ top-k retrieve ──▶ llama3.1:8b ──▶ answer + cited sources
                                            [Ollama, on the host GPU]
```

## Prerequisites

- Docker Desktop (Windows or macOS) or Docker Engine with the compose
  plugin (Linux/WSL2)
- [Ollama](https://ollama.com) running on the host, with two models:

  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b
  ```

Ollama stays on the host rather than in compose because it owns the
GPU; the containers only need HTTP access to it.

## Setup

```bash
cp .env.example .env        # optional: defaults work out of the box
docker compose up --build -d
```

First boot waits for Ollama and Chroma, ingests `docs/`, then serves on
port 8000. Watch it happen:

```bash
docker compose logs -f rag-api
```

Verify, then ask the demo question (the bundled
[`docs/example.md`](docs/example.md) makes this work immediately):

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

The answer cites its sources by block number, and the `sources` array
carries file, chunk index, similarity score, and a text preview for
each one.

## Usage

| Method and path | Purpose |
|---|---|
| `POST /ask` | `{"question": "...", "top_k": 4}` (top_k optional) |
| `GET /health` | Dependency-aware status; `503` when degraded |
| `POST /ingest/refresh` | Re-index `docs/` without a restart |
| `GET /docs` | Interactive OpenAPI docs (FastAPI built-in) |

Add your own documents to `docs/`, then either restart or:

```bash
curl -sS -X POST http://localhost:8000/ingest/refresh
```

Ingest is idempotent by content hash: unchanged files are skipped,
edited files have their stale chunks deleted and replaced, and nothing
ever duplicates. Note that `.gitignore` excludes `docs/*` except the
example file, so private documents stay out of the public repo by
default.

Tuning lives in `.env`; every knob and its default is listed in
[`.env.example`](.env.example) and typed in
[`app/config.py`](app/config.py).

## Networking: how containers reach host Ollama

The containers call `http://host.docker.internal:11434`. What makes
that name resolve differs by platform:

| Where Docker runs | Where Ollama runs | Works because |
|---|---|---|
| Docker Desktop (Windows) | Windows host | Docker Desktop provides `host.docker.internal` natively |
| Docker Desktop (WSL2 backend) | Windows host | Same; Desktop routes to the Windows host |
| Docker Engine in WSL2 | WSL2 | `extra_hosts: host-gateway` in the compose file maps the name |
| Docker Engine on Linux | Same machine | Same `host-gateway` mapping |

**The Windows gotcha.** Ollama on Windows binds to `127.0.0.1` by
default, which containers cannot reach even when the hostname resolves.
Fix it once on the host:

```powershell
setx OLLAMA_HOST 0.0.0.0
```

then quit and restart Ollama from the system tray. The symptom of
skipping this is the API logging `Waiting for Ollama (attempt n/30)`
thirty times and exiting with a pointer to this section.

## How it fits into Atlas Systems

This kit is the orchestration-layer sibling of
[`atlas-kit-python-rag`](https://github.com/AtlasReaper311/atlas-kit-python-rag):
that repo proves the pipeline as a tested, CI-green Python library; this
one proves the same architecture as a deployable multi-container
service. It also closes the docker-compose gap in the Atlas decisions
log, and the pipeline shape (ingest, embed, retrieve, generate) is the
one the honours project will reuse against UE5 audio documentation.

Optional event-bus hookup: set `NOTIFY_URL` and `NOTIFY_TOKEN` in
`.env` and the service reports ingest errors and startup to
[`atlas-notify`](https://github.com/AtlasReaper311/atlas-notify). Both
empty by default; the kit stands alone.
