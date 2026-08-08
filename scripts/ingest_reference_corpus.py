#!/usr/bin/env python3
"""Ingest OpenStax Psychology 2e chapters into a separate, cited reference corpus.

Source and license
-------------------
OpenStax Psychology 2e (Rice University), verified directly from the book's own
preface: "licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
4.0 (CC BY NC-SA) license... you can non-commercially distribute, remix, and
build upon the content, as long as you provide attribution to OpenStax and its
content contributors, and distribute all derivatives under the same license."
https://openstax.org/books/psychology-2e/pages/preface

That means two hard rules, enforced by this script and everything downstream:

1. The ingested text and its embeddings are NEVER committed to this repo. They
   live in .runtime/ (gitignored, same treatment as every other generated or
   third-party artifact here) and in the local Chroma store (also gitignored).
2. Every citation surfaced to the user carries OpenStax's required attribution:
   "Access for free at <the exact page URL>". See app.py's reference-sources
   response field and templates/index.html's reference-citation rendering.

Chapters ingested (chosen for relevance to journaling/reflection, not the whole
19-chapter book): 10 Motivation and Emotion, 11 Personality, 14 Stress
Lifestyle and Health, 16 Therapy and Treatment. Verified via web search against
the book's actual table of contents, not guessed.

Why HTML scraping, not a PDF
-----------------------------
OpenStax serves each section as its own stable, server-rendered HTML page
(verified: curling a section URL returns the real paragraph text, not a JS
shell). That's easier to chunk cleanly than a PDF and needs no PDF-parsing
dependency -- only beautifulsoup4 (optional-only, requirements-optional.txt).

Section discovery: a chapter's introduction page has its "Chapter Outline" as
plain text (e.g. "14.1 What Is Stress? 14.2 Stressors..."), which is parsed for
(chapter, section, title) tuples. The section URL slug is then a naive
lowercase-and-hyphenate of the title. This is NOT assumed correct: after
fetching, the page's own <title> tag is checked against the expected section
title, and a mismatch is skipped with a warning rather than silently ingesting
the wrong page under the wrong citation.

Run:
    PYTHONPATH=. python scripts/ingest_reference_corpus.py            # all chapters
    PYTHONPATH=. python scripts/ingest_reference_corpus.py --chapter 14  # one chapter
    PYTHONPATH=. python scripts/ingest_reference_corpus.py --dry-run   # fetch+chunk, don't write to Chroma
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ingest_reference")

BASE_URL = "https://openstax.org/books/psychology-2e/pages"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".runtime" / "reference_corpus"

LICENSE = "CC BY-NC-SA 4.0"
SOURCE_BOOK = "OpenStax Psychology 2e"

# Verified against the book's real table of contents (web search + a fetched
# chapter-introduction page), not guessed. Chosen for relevance to a
# journaling/reflection app, not exhaustive coverage of the book.
CHAPTERS = [
    (10, "Motivation and Emotion"),
    (11, "Personality"),
    (14, "Stress, Lifestyle, and Health"),
    (16, "Therapy and Treatment"),
]

# Chunking target. Sections run ~5k-25k characters of prose; grouping
# consecutive paragraphs up to this budget keeps each chunk a coherent, roughly
# journal-entry-sized unit rather than one giant per-section blob that would
# dominate retrieval scoring and blow past a reasonable prompt context.
CHUNK_TARGET_CHARS = 1200

# `|$` matters: the outline substring is sliced to end right before the
# chapter's own "Introduction" heading (see discover_sections), so the LAST
# section in any chapter has nothing after it for the lookahead to match
# against except end-of-string. Without it, every chapter silently lost its
# final section -- found by dry-running chapter 14 and getting 4 sections
# instead of the verified 5 (14.5 "The Pursuit of Happiness" went missing).
_OUTLINE_ENTRY = re.compile(r"(\d+)\.(\d+)\s+([^0-9]+?)(?=\s+\d+\.\d+\s|\s+Introduction\b|$)")
_REQUEST_HEADERS = {"User-Agent": "ai-health-journal-reference-ingest/1.0"}


@dataclass(frozen=True)
class Section:
    chapter: int
    section: str  # "14.1"
    title: str
    url: str


@dataclass(frozen=True)
class Chunk:
    text: str
    section: Section
    chunk_index: int


def _fetch(url: str, cache_name: str) -> str:
    """Fetch a URL with a local cache, so repeat runs don't hit the network."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    log.info("  fetching %s", url)
    request = urllib.request.Request(url, headers=_REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
        html = response.read().decode("utf-8", errors="replace")
    cache_path.write_text(html, encoding="utf-8")
    time.sleep(0.5)  # polite delay between real requests
    return html


def discover_sections(chapter: int, title: str) -> List[Section]:
    """Parse a chapter's introduction page for its real section list."""
    from bs4 import BeautifulSoup

    url = f"{BASE_URL}/{chapter}-introduction"
    html = _fetch(url, f"ch{chapter}_introduction.html")
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main", class_="page-content")
    if main is None:
        log.warning("  chapter %d: no main-content found, skipping", chapter)
        return []

    text = main.get_text(" ", strip=True)
    outline_start = text.find("Chapter Outline")
    if outline_start == -1:
        log.warning("  chapter %d: no 'Chapter Outline' text found, skipping", chapter)
        return []
    # Outline block ends where the "Introduction" prose begins.
    outline_end = text.find(" Introduction ", outline_start)
    outline_text = text[outline_start:outline_end if outline_end != -1 else outline_start + 600]

    sections: List[Section] = []
    for match in _OUTLINE_ENTRY.finditer(outline_text):
        num_chapter, num_section, section_title = match.groups()
        section_title = section_title.strip().rstrip(".")
        slug_title = re.sub(r"[^a-z0-9]+", "-", section_title.lower()).strip("-")
        slug = f"{num_chapter}-{num_section}-{slug_title}"
        sections.append(
            Section(
                chapter=int(num_chapter),
                section=f"{num_chapter}.{num_section}",
                title=section_title,
                url=f"{BASE_URL}/{slug}",
            )
        )
    log.info("  chapter %d (%s): found %d sections", chapter, title, len(sections))
    return sections


def fetch_section_text(section: Section) -> Optional[List[str]]:
    """Fetch one section, verify it's the page we think it is, return paragraphs.

    Returns None (and logs a warning) rather than raising, so one bad slug in a
    chapter doesn't abort the whole ingestion run.
    """
    from bs4 import BeautifulSoup

    cache_name = f"{section.section.replace('.', '_')}.html"
    try:
        html = _fetch(section.url, cache_name)
    except Exception as exc:  # noqa: BLE001 - reported, ingestion continues
        log.warning("  %s: fetch failed (%s), skipping", section.url, type(exc).__name__)
        return None

    soup = BeautifulSoup(html, "html.parser")
    page_title = (soup.title.get_text() if soup.title else "") or ""
    if section.title.lower() not in page_title.lower():
        log.warning(
            "  %s: title mismatch (expected %r, got %r), skipping -- slug guess was wrong",
            section.url, section.title, page_title,
        )
        return None

    main = soup.find("main", class_="page-content")
    if main is None:
        log.warning("  %s: no main-content, skipping", section.url)
        return None

    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    return [p for p in paragraphs if len(p) > 20]  # drop stray one-word fragments


def chunk_paragraphs(paragraphs: List[str], section: Section) -> List[Chunk]:
    """Group consecutive paragraphs into ~CHUNK_TARGET_CHARS chunks."""
    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_len = 0
    for para in paragraphs:
        if buffer and buffer_len + len(para) > CHUNK_TARGET_CHARS:
            chunks.append(Chunk(text=" ".join(buffer), section=section, chunk_index=len(chunks)))
            buffer, buffer_len = [], 0
        buffer.append(para)
        buffer_len += len(para)
    if buffer:
        chunks.append(Chunk(text=" ".join(buffer), section=section, chunk_index=len(chunks)))
    return chunks


def build_all_chunks(chapters: List[int]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for chapter, title in CHAPTERS:
        if chapters and chapter not in chapters:
            continue
        for section in discover_sections(chapter, title):
            paragraphs = fetch_section_text(section)
            if not paragraphs:
                continue
            section_chunks = chunk_paragraphs(paragraphs, section)
            all_chunks.extend(section_chunks)
            log.info(
                "    %s %s -> %d paragraphs, %d chunks",
                section.section, section.title, len(paragraphs), len(section_chunks),
            )
    return all_chunks


def write_to_chroma(chunks: List[Chunk], namespace: str) -> int:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from config import load_config
    from vector_store.embeddings import build_embedding_function
    from vector_store.chroma_store import ChromaStore

    cfg = load_config()
    store = ChromaStore(embedding_function=build_embedding_function(cfg))

    # Rebuild the namespace clean each run rather than upserting -- ingestion is
    # cheap (cached HTML) and idempotent-by-rebuild is simpler and more
    # predictable than reasoning about partial re-runs with changed chunking.
    store.clear_namespace(namespace)

    written = 0
    for chunk in chunks:
        entry_id = f"ref_{chunk.section.section.replace('.', '_')}_{chunk.chunk_index}"
        ok = store.add_entry(
            entry_id=entry_id,
            text=chunk.text,
            metadata={
                "kind": "reference_passage",
                "source_book": SOURCE_BOOK,
                "license": LICENSE,
                "chapter": chunk.section.chapter,
                "section": chunk.section.section,
                "section_title": chunk.section.title,
                "source_url": chunk.section.url,
                "attribution": f"Access for free at {chunk.section.url}",
            },
            namespace=namespace,
        )
        written += int(ok)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int, action="append", default=[], help="limit to one chapter (repeatable)")
    parser.add_argument("--namespace", default="reference:psychology")
    parser.add_argument("--dry-run", action="store_true", help="fetch and chunk only, don't write to Chroma")
    args = parser.parse_args()

    log.info("=== Reference corpus ingestion: %s (%s) ===", SOURCE_BOOK, LICENSE)
    log.info("  cache dir: %s (gitignored)", CACHE_DIR)
    log.info("  mode: %s\n", "DRY RUN (fetch + chunk only)" if args.dry_run else "WRITE to Chroma")

    chunks = build_all_chunks(args.chapter)
    total_chars = sum(len(c.text) for c in chunks)
    log.info("\n  total chunks: %d  (%d chars, avg %d/chunk)",
              len(chunks), total_chars, total_chars // max(1, len(chunks)))

    if not chunks:
        log.error("No chunks produced. Nothing to write.")
        return 1

    if args.dry_run:
        log.info("\nDry run complete. Nothing written to Chroma.")
        return 0

    written = write_to_chroma(chunks, args.namespace)
    log.info("\nWrote %d/%d chunks to namespace %r.", written, len(chunks), args.namespace)
    return 0 if written == len(chunks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
