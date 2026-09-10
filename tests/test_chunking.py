from __future__ import annotations

import pytest

from vector_store.chunking import ChunkingConfig, chunk_text


class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello journal")
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].text == "hello journal"

    def test_long_text_multiple_chunks(self):
        cfg = ChunkingConfig(chunk_size=40, chunk_overlap=10)
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = chunk_text(text, cfg)
        assert len(chunks) >= 2
        assert [c.index for c in chunks] == list(range(len(chunks)))
        # Overlap means later chunks share some content region with prior ones
        assert all(c.text for c in chunks)

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=10, chunk_overlap=0)
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)
