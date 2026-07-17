# Privacy and the trust boundary

This project turns journal history into a locally stored RAG index. The design
goal: **no journal content leaves the machine** unless you explicitly open a
cloud gate. This document states what that guarantees, and what it does not.

## Trust boundary

The trust boundary is **local disk**. Journal text, its embeddings, and the
vector index all live on the machine running the app. Nothing about an entry is
transmitted off-host under the default and recommended configuration.

## Data flow (default config)

```
journal entry
  -> Flask /analyze (in-process, localhost)
  -> Ollama at localhost:11434            (analysis; local inference)
  -> local embedder (ONNX MiniLM, on-CPU) (embeddings; no network)
  -> Chroma PersistentClient ./storage/chroma  (index; local disk)
```

Retrieval reverses the last two steps: the current entry is embedded locally and
matched against the local Chroma collection for the session namespace.

## The three leak points, and how each is closed

A RAG journal can leak content in three independent places. "Local vector DB" is
not enough on its own.

| Layer | Default | Content leaves host? | Gate |
|---|---|---|---|
| Vector DB | Chroma, `./storage/chroma`, telemetry off | No | `VECTOR_BACKEND=chroma`, `ALLOW_CLOUD_VECTORSTORE=false` |
| Embeddings | Chroma built-in ONNX MiniLM, on-CPU | No | none needed (local by construction) |
| LLM inference | Ollama `localhost:11434` | No | `LLM_BACKEND=ollama`, `ALLOW_CLOUD_LLM=false` |
| Transcription | local Whisper | No | optional dependency, runs on-host |

Cloud paths exist but are gated shut by default:

- **Pinecone** (cloud vector DB): requires `ALLOW_CLOUD_VECTORSTORE=true`. If
  enabled, entry vectors and text are stored off-host. Embeddings are still
  computed locally (SentenceTransformer MiniLM).
- **Anthropic** (cloud LLM): double-gated. Requires `LLM_BACKEND=anthropic`
  **and** `ALLOW_CLOUD_LLM=true`. The Anthropic client is never constructed and
  `ANTHROPIC_API_KEY` is never read into memory while the gate is closed
  (`providers/factory.py`). If enabled, raw entry text is sent to Anthropic.

## At rest

- Entries are stored in the local Chroma DB. Under `PRIVACY_MODE=strict`, emails
  and phone numbers are scrubbed (`privacy/redact.py`) before an entry is
  persisted, so the retrievable history and its metadata carry no such PII. Mode
  change is not retroactive: entries written under a looser mode keep their text.
- `PRIVACY_MODE=balanced` (default) stores raw entry text locally. This is the
  historical behavior; local disk is the trust boundary either way.
- `chat_history.json` (raw source chats) and the Chroma DB are gitignored and
  never committed.

## What "private" guarantees — and what it does not

Guarantees (default config):

- No journal entry text, embedding, or derived insight is transmitted off-host.
- No third-party telemetry from the vector store (`anonymized_telemetry=False`).

Does **not** guarantee:

- Encryption at rest. The local DB is plaintext on disk; rely on full-disk
  encryption for that layer.
- Redaction unless `PRIVACY_MODE=strict`. Balanced mode stores raw text locally.
- Offline-on-first-run: model weights (Chroma embedder ~80MB, Whisper) download
  once from a public CDN. Pre-cache them for a fully air-gapped run. These are
  model weights inbound, never journal data outbound.

## How to verify

- Retrieval path, live: `make verify-rag` (`scripts/verify_rag.py`) proves real
  semantic match, namespace isolation, and self-exclusion against live Chroma.
- Gates: confirm `.env` has `ALLOW_CLOUD_VECTORSTORE=false`, `LLM_BACKEND=ollama`,
  `ALLOW_CLOUD_LLM=false`.
- At-rest redaction: `pytest tests/test_privacy_redaction.py`.
