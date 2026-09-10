"""Configurable text chunking for the journal ingestion path.

Chunk size and overlap live in config so retrieval sweeps can vary them
without code changes. Defaults are conservative for short journal entries;
longer reflective writing benefits from smaller chunks + overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 400
    chunk_overlap: int = 80

    def __post_init__(self) -> None:
        if self.chunk_size < 32:
            raise ValueError("chunk_size must be >= 32")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")


def chunk_text(text: str, cfg: ChunkingConfig | None = None) -> List[Chunk]:
    """Split ``text`` into overlapping character windows.

    Short texts that fit in one window return a single chunk (index 0).
    Empty / whitespace-only input returns an empty list.
    """
    cfg = cfg or ChunkingConfig()
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if len(cleaned) <= cfg.chunk_size:
        return [Chunk(index=0, text=cleaned, start_char=0, end_char=len(cleaned))]

    step = cfg.chunk_size - cfg.chunk_overlap
    chunks: List[Chunk] = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(start + cfg.chunk_size, len(cleaned))
        # Prefer breaking on whitespace near the window end when possible.
        if end < len(cleaned):
            soft = cleaned.rfind(" ", start + cfg.chunk_overlap, end)
            if soft > start:
                end = soft
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                Chunk(index=index, text=piece, start_char=start, end_char=end)
            )
            index += 1
        if end >= len(cleaned):
            break
        start = max(end - cfg.chunk_overlap, start + step)
    return chunks
