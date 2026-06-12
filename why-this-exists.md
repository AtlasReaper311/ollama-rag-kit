# Why this exists

## Context

The stack already had a RAG codebase: `atlas-kit-python-rag`, a tested
library with green CI proving the pipeline logic works. What it does
not prove is operational capability: that the same architecture can be
packaged, orchestrated, and handed to someone who types one command and
gets a working service. Those are different skills, and the Atlas
decisions log explicitly listed multi-container orchestration as a gap.

There is also a practical driver. Ramone's document collection (project
notes, case study drafts, hardware records) lives on SPECULAR-CORE, and
questions about it should be answerable without sending a single byte
to a hosted API.

## Options considered

**Extend `atlas-kit-python-rag` with a compose file.** Tempting, but it
muddies what each repo demonstrates. The library repo's value is "clean
Python, typed, tested"; bolting deployment onto it weakens that signal
and produces one repo doing two jobs averagely.

**Cloud-hosted RAG (OpenAI embeddings plus a managed vector store).**
Less infrastructure to run, but it breaks the £0 constraint at any real
usage, couples the system to vendor uptime, and forfeits the privacy
property entirely. For a kit whose corpus is private notes, sending the
corpus away defeats the point.

**Local, containerised, Ollama-backed.** Embeddings and generation on
hardware already owned, vector store pinned and persistent, the whole
thing reproducible from one compose file. Cost: slower generation than
a frontier API and the operational burden of running it. Both
acceptable; the burden is the demonstration.

## Decision

A two-container compose stack (pinned `chromadb/chroma:0.5.23` plus a
FastAPI app on `python:3.12-slim`) with Ollama deliberately left on the
host, because the host owns the GPU and the containers only need HTTP.
`extra_hosts: host-gateway` makes the same compose file work on Docker
Desktop and native Linux without edits.

Three implementation choices carry most of the weight. Ingest is
idempotent by content hash, so container restarts are safe and edits
replace rather than accumulate. The application owns its readiness with
retry loops instead of leaning on compose healthchecks, so the same
image behaves identically under any orchestrator. And `num_ctx` is set
explicitly to 4096, because Ollama's 2048 default silently truncates
exactly the retrieved context a RAG system depends on; that one is a
gotcha worth the config line and the comment.

## Consequences

The stack boots from nothing with `docker compose up --build` and
answers a demo question with citations inside a minute of model
warm-up. The corpus never leaves the machine. The cost is local-model
answer quality and a service to keep running, both of which are the
point rather than the price.

The transferable principle: the RAG pipeline shape (ingest, embed,
retrieve with provenance, generate grounded) is identical at every
scale. Swap ChromaDB for a managed vector store and Ollama for a hosted
API and the architecture does not change; build it small and local
first, and the big version is a configuration exercise.
