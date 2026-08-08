"""Tests for scripts/ingest_reference_corpus.py's pure logic.

Network-free: these test the regex that parses a chapter's outline text and the
paragraph-chunking logic, not the live fetch. The outline regex earned this
suite the hard way -- see test_last_outline_entry_is_not_dropped, a regression
for a bug found by dry-running chapter 14 and getting 4 sections instead of 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.ingest_reference_corpus import (  # noqa: E402
    _OUTLINE_ENTRY,
    Section,
    chunk_paragraphs,
)


class TestOutlineRegex:
    def test_parses_a_full_chapter_outline(self):
        text = "Chapter Outline 14.1 What Is Stress? 14.2 Stressors 14.3 Stress and Illness"
        matches = [(m.group(1), m.group(2), m.group(3).strip()) for m in _OUTLINE_ENTRY.finditer(text)]
        assert matches == [
            ("14", "1", "What Is Stress?"),
            ("14", "2", "Stressors"),
            ("14", "3", "Stress and Illness"),
        ]

    def test_last_outline_entry_is_not_dropped(self):
        """Regression: the lookahead required a following section number or the
        word "Introduction" to close a match. When the outline substring is
        sliced to end right before "Introduction" (as discover_sections does),
        the last entry has nothing left to match against and silently
        vanished -- chapter 14 dry-ran to 4 sections instead of the verified 5.
        """
        text = "Chapter Outline 14.1 What Is Stress? 14.5 The Pursuit of Happiness"
        matches = [m.group(3).strip() for m in _OUTLINE_ENTRY.finditer(text)]
        assert matches == ["What Is Stress?", "The Pursuit of Happiness"]

    def test_single_entry_outline(self):
        text = "Chapter Outline 11.1 What Is Personality?"
        matches = [m.group(3).strip() for m in _OUTLINE_ENTRY.finditer(text)]
        assert matches == ["What Is Personality?"]


class TestChunkParagraphs:
    SECTION = Section(chapter=14, section="14.1", title="What Is Stress?", url="https://example.com")

    def test_short_section_becomes_one_chunk(self):
        paragraphs = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        chunks = chunk_paragraphs(paragraphs, self.SECTION)
        assert len(chunks) == 1
        assert chunks[0].text == "First paragraph. Second paragraph. Third paragraph."

    def test_long_section_splits_at_the_target_budget(self):
        from scripts.ingest_reference_corpus import CHUNK_TARGET_CHARS

        big_para = "x" * (CHUNK_TARGET_CHARS - 10)
        paragraphs = [big_para, big_para, big_para]
        chunks = chunk_paragraphs(paragraphs, self.SECTION)
        # Each paragraph alone is near the budget, so a second paragraph always
        # pushes over it -> one paragraph per chunk.
        assert len(chunks) == 3
        assert all(c.text == big_para for c in chunks)

    def test_chunks_never_lose_or_duplicate_content(self):
        paragraphs = [f"Paragraph number {i} with some real content in it." for i in range(20)]
        chunks = chunk_paragraphs(paragraphs, self.SECTION)
        reassembled = " ".join(c.text for c in chunks)
        for para in paragraphs:
            assert para in reassembled

    def test_empty_paragraph_list_produces_no_chunks(self):
        assert chunk_paragraphs([], self.SECTION) == []

    def test_chunk_index_is_sequential(self):
        big_para = "y" * 1000
        chunks = chunk_paragraphs([big_para, big_para, big_para], self.SECTION)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
