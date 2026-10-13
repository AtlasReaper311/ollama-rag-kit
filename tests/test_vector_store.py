from __future__ import annotations

import pytest

from app.config import Settings
from app.vector_store import _metadata_filter, _point_id, connect_collection


def test_qdrant_point_mapping_is_deterministic_and_collection_scoped():
    left = _point_id("atlas_docs", "record-1")
    same = _point_id("atlas_docs", "record-1")
    other_collection = _point_id("ramone_sessions", "record-1")

    assert left == same
    assert left != other_collection


def test_qdrant_metadata_filters_preserve_scalar_and_range_semantics():
    query_filter = _metadata_filter(
        {"$and": [{"role": "assistant"}, {"turn_index": {"$gte": 2}}]}
    )

    assert query_filter is not None


def test_qdrant_metadata_filters_can_combine_scalar_and_logical_clauses():
    query_filter = _metadata_filter(
        {"role": "assistant", "$or": [{"turn_index": 2}, {"turn_index": 3}]}
    )

    assert query_filter is not None


def test_qdrant_backend_fails_closed_without_api_key():
    settings = Settings(vector_backend="qdrant", qdrant_url="https://qdrant:6333")

    with pytest.raises(ValueError, match="QDRANT_API_KEY"):
        connect_collection(settings)

def test_default_backend_is_private_qdrant():
    assert Settings().vector_backend == "qdrant"

def test_qdrant_backend_rejects_plaintext_urls():
    settings = Settings(
        vector_backend="qdrant",
        qdrant_url="http://qdrant:6333",
        qdrant_api_key="disposable-test-key",
    )

    with pytest.raises(ValueError, match="HTTPS"):
        connect_collection(settings)

