"""Typed configuration for the RAG service.

Every tunable lives here and is overridable from the environment, which
keeps docker-compose.yml the single place where deployment values are
set. pydantic-settings validates types at startup, so a typo like
CHUNK_SIZE=12oo fails the boot loudly instead of corrupting the index
quietly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama runs outside compose on the host, so the default targets the
    # host gateway. docker-compose.yml maps host.docker.internal on Linux
    # too via extra_hosts, making this default work on every platform.
    ollama_host: str = "http://host.docker.internal:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"

    # Low temperature: a RAG answer should restate the documents, not
    # improvise around them.
    temperature: float = 0.2

    # Ollama defaults to a 2048-token context window, which four chunks
    # plus the system prompt can overflow silently (the model just stops
    # seeing the earliest context). 4096 gives comfortable headroom.
    num_ctx: int = 4096

    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    collection_name: str = "atlas_docs"

    docs_dir: str = "/srv/docs"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embed_batch_size: int = 32
    top_k: int = 4

    # Generation is the slow call by far; embeddings are quick but batch
    # size makes them variable. Health checks stay snappy.
    embed_timeout_seconds: float = 120.0
    generate_timeout_seconds: float = 300.0
    health_timeout_seconds: float = 5.0

    # Optional integration with atlas-notify. Both empty by default so the
    # kit stands alone; set NOTIFY_URL and NOTIFY_TOKEN to wire it in.
    notify_url: str = ""
    notify_token: str = ""
    notify_on_start: bool = False

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    lru_cache makes this a lazy singleton: the environment is read once,
    and every module sees the same validated object.
    """
    return Settings()
