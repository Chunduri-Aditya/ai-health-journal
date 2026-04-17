# Upgrade 02 — Replace Cookie Session History with SQLite Persistence

## Problem

History and analysis results live entirely in the Flask session:

```356:363:app.py
def _append_to_session(entry: str, response: str, analysis_json: Optional[Dict]) -> None:
    if "chat" not in session:
        session["chat"] = []
    item: Dict[str, Any] = {"entry": entry, "response": response}
    if analysis_json is not None:
        item["analysis_json"] = analysis_json
    session["chat"].append(item)
    session.modified = True
```

Flask's default session is **cookie-backed** (signed + client-side). Consequences already present today:

- Each journal entry + full analysis JSON is serialized into a cookie → easily exceeds the 4 KB practical limit after a few entries.
- `app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)` at `app.py:61` means in dev the key rotates on every restart → cookies silently invalidate → "durable" history is not durable.
- No server-side record means no way to audit which retrieval sources informed which analysis, no export, no deletion, no memory across sessions.

## Goal

All entries, analyses, and retrieval provenance persist in a **local SQLite database**. Flask session holds only a lightweight `session_id` pointer. History survives restarts. A reset deletes rows for the active session, not a cookie key.

## Dependencies

- **Benefits from:** `01-retrieval-grounding.md` (structured sources → populated `retrieval_hits` rows).
- **Benefits from:** `06-tests-schema.md` (storage repository tests).
- **Safe to parallelize with:** `04-transcription.md`, `05-sse-streaming.md`.

## Plan

### A. Module layout

```
storage/
  __init__.py
  db.py              # connection factory, migration runner
  migrations/
    001_init.sql     # initial schema (DDL below)
  repository.py      # domain-level API used by app.py
  models.py          # dataclasses / typed dicts for rows
```

### B. Schema (`storage/migrations/001_init.sql`)

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    label        TEXT
);

CREATE TABLE IF NOT EXISTS entries (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    body         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_session_created
    ON entries(session_id, created_at);

CREATE TABLE IF NOT EXISTS analyses (
    id                 TEXT PRIMARY KEY,
    entry_id           TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    mode               TEXT NOT NULL CHECK(mode IN ('legacy','baseline_json','quality')),
    generator_model    TEXT,
    verifier_model     TEXT,
    fallback_model     TEXT,
    analysis_json      TEXT,          -- serialized JSON
    rendered_text      TEXT NOT NULL,
    groundedness_score REAL,
    rewrite_applied    INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_entry ON analyses(entry_id);

CREATE TABLE IF NOT EXISTS retrieval_hits (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id      TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    source_entry_id  TEXT REFERENCES entries(id) ON DELETE SET NULL,
    rank             INTEGER NOT NULL,
    score            REAL NOT NULL,
    snippet          TEXT NOT NULL,
    metadata_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_hits_analysis ON retrieval_hits(analysis_id);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
```

Ids: use ULIDs or UUIDv7 (lexicographically time-sortable). Stick with strings.

### C. `storage/db.py`

```python
from __future__ import annotations
import sqlite3, os, pathlib, logging
from contextlib import contextmanager

_DB_PATH = os.getenv("AIHJ_DB_PATH", "./storage/aihj.sqlite3")
_MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, isolation_level=None)  # autocommit-ish
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    pathlib.Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    current_version = 0
    if exists:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_version"
        ).fetchone()
        current_version = int(row["version"])
    # Run all .sql files in sorted order with version > current_version.
    # Record each version in schema_version inside the same transaction.
    ...

@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()
```

### D. `storage/repository.py`

Domain API consumed by `app.py`:

```python
def ensure_session(session_id: str) -> None: ...
def create_entry(session_id: str, body: str) -> Entry: ...
def create_analysis(
    entry: Entry,
    *,
    mode: str,
    generator_model: str,
    verifier_model: Optional[str],
    fallback_model: Optional[str],
    analysis_json: Optional[Dict[str, Any]],
    rendered_text: str,
    groundedness_score: Optional[float],
    rewrite_applied: bool,
    sources: List[Dict[str, Any]],
) -> Analysis: ...
def list_history(
    session_id: str,
    *,
    limit: int = 100,
    before_created_at: Optional[str] = None,
) -> List[HistoryItem]: ...
def reset_session(session_id: str) -> int: ...  # returns rows deleted
def delete_entry(entry_id: str) -> None: ...
def export_session(session_id: str) -> Dict[str, Any]: ...
```

`create_analysis` writes the analysis row plus one `retrieval_hits` row per source in a single transaction.

### E. `app.py` changes

1. Replace `secrets.token_hex(32)` fallback with a persisted secret:
   - If `SECRET_KEY` env unset, read/write `./storage/secret.key` on first boot.
   - Warn once at startup when falling back.

2. Session becomes pointer-only:

   ```python
   def _ensure_session_id() -> str:
       sid = session.get("sid")
       if not sid:
           sid = new_ulid()
           session["sid"] = sid
           session.permanent = True
           repository.ensure_session(sid)
       return sid
   ```

3. `_append_to_session` → `repository.create_entry` + `repository.create_analysis`.

4. `/session/history` → `repository.list_history(sid)`.

5. `/session/reset` → `repository.reset_session(sid)` then clear the cookie key.

6. New endpoints:

   | Method | Path                          | Action                                     |
   |--------|-------------------------------|--------------------------------------------|
   | GET    | `/session/export`             | Full export (JSON) of current session      |
   | DELETE | `/session/entries/<entry_id>` | Delete one entry                           |
   | POST   | `/session/reset?purge=true`   | Also drop the session row itself           |

7. Init: call `storage.db.init_db()` once at startup.

### F. Frontend

`templates/index.html` already calls `/session/history` and `/session/reset`. Response shape stays a list of `{entry, response, analysis_json?}` objects for backward compatibility. Add an optional `sources` field surfaced through 01.

Add pagination support up front so the route shape does not need to change again after users accumulate larger histories:

- `GET /session/history?limit=50`
- `GET /session/history?limit=50&before=<ISO8601 timestamp>`

### G. Data retention

- Config knobs: `HISTORY_RETENTION_DAYS` (default `0` = never auto-delete), `HISTORY_MAX_ENTRIES_PER_SESSION` (default `0` = unbounded).
- A background sweep is not required for v1; only act on explicit user delete + `/session/reset`.

### H. Redaction boundary

Do **not** silently mutate the primary journal record stored in SQLite. The database is the user's canonical local journal history and should preserve what they wrote.

If redaction is enabled later (see 07), apply it:

- before writing to the vector store used for retrieval, and
- optionally on export paths designed for sharing,

but not on the canonical `entries.body` field by default.

### I. Migration from existing users

No migration. Prior data lived in cookies only and is already ephemeral. The first post-upgrade request creates a fresh session.

## New / changed interfaces

### Config (new env vars)

| Env var                              | Default                          | Purpose                                  |
|--------------------------------------|----------------------------------|------------------------------------------|
| `AIHJ_DB_PATH`                       | `./storage/aihj.sqlite3`         | SQLite path                              |
| `SECRET_KEY_FILE`                    | `./storage/secret.key`           | Fallback key file                        |
| `HISTORY_RETENTION_DAYS`             | `0` (off)                        | Auto-delete threshold                    |
| `HISTORY_MAX_ENTRIES_PER_SESSION`    | `0` (unbounded)                  | Cap per session                          |

### Endpoints (new)

- `GET /session/export` → `{ session: {...}, entries: [{entry, analyses: [...]}] }`
- `DELETE /session/entries/<entry_id>` → `204 No Content`
- `GET /session/history?limit=<n>&before=<ISO8601>` → paginated history list

## Acceptance criteria

1. Restart the app. History from before the restart is still visible on the same browser.
2. Insert 50+ entries; cookie size stays small; no "cookie too large" warnings.
3. `POST /session/reset` deletes corresponding rows from `entries` and `analyses` (verify via `sqlite3 storage/aihj.sqlite3 '.tables'` then `SELECT COUNT`).
4. `/session/export` returns a valid JSON document that can round-trip back via a future import path (out of scope for this upgrade).
5. Migration runner is idempotent: running the app twice does not re-apply `001_init.sql`.
6. Storage layer has pytest coverage (from 06) for create/list/reset/delete.
7. DB file permissions: `0600`. Parent directory auto-created.
8. `DELETE /session/entries/<entry_id>` removes the entry and its analyses but preserves unrelated session history.
9. `GET /session/history?limit=10&before=...` returns stable reverse-chronological pagination without duplicating rows across pages.

## Risks & open questions

- **Concurrency.** Flask dev server is single-threaded by default, but `FLASK_RUN_THREADS` or production WSGI runners aren't. SQLite WAL mode handles multi-reader/single-writer; still, wrap writes in `with get_conn() as c, c:` to take an implicit transaction.
- **Backup / portability.** A plain `.sqlite3` file is trivially portable. Document that in the README.
- **Encryption at rest.** Out of scope; if added later, consider `sqlcipher3` (but adds a compiled dep that cuts against the minimal-install posture).
- **Multi-user deployments.** If this app is ever deployed with more than one local user, `session_id` alone is not an auth boundary. Flag this in docs; do not silently suggest otherwise.
- **Privacy module wiring (07 crossover).** `privacy/redact.py` is a better fit for vector-store writes and export/share paths than for mutating the canonical `entries.body` field. Keep that boundary explicit if 07 lands later.
- **Secret key file permissions.** Apply `0600` to `storage/secret.key` as well as the SQLite database so local account boundaries are not weakened by world-readable files.

## Touch list

- `app.py` — session handling, route bodies, startup hook.
- `config.py` — new env vars.
- `storage/__init__.py` — new.
- `storage/db.py` — new.
- `storage/repository.py` — new.
- `storage/models.py` — new.
- `storage/migrations/001_init.sql` — new.
- `requirements-core.txt` — no new deps (sqlite3 is stdlib); confirm.
- `.gitignore` — add `storage/aihj.sqlite3*`, `storage/secret.key`.
- `README.md` — persistence section + backup guidance.
- `Makefile` — optional `make db-reset` target.
- `tests/storage/` — repository tests (under 06).
