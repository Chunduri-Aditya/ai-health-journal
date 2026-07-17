# Privacy Modules

This directory contains privacy-related helpers that are only partially integrated today.

Current status:

- `redact.py` - intended to provide text redaction utilities for retrieval, export, or future sharing flows.
- `local_text_cache.py` - legacy local cache helper that is a candidate for removal once SQLite-backed persistence replaces its remaining use cases.

Boundary to preserve:

- The local database should remain the canonical journal record unless the user explicitly opts into full-storage redaction.
- Redaction is more naturally applied to retrieval/indexing copies and export/share paths than to the primary stored entry text.

If runtime behavior changes:

1. Update this README.
2. Update `docs/upgrades/07-cleanup.md`.
3. Add or update tests that pin the chosen redaction boundary.
