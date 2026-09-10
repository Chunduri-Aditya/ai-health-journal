# Journal Agent — Agentic RAG Service

**Product surface** for this repository: the FastAPI + LangGraph + pgvector service.
The Flask UI (`./start.sh`) is a **local lab / reference** for privacy-first journaling experiments — not the deployed product.

## Deployment posture (single-tenant)

- One owner behind one shared `JOURNAL_AGENT_API_KEY`. Outside `ENV=dev`, the key is **required** (fail-closed).
- `namespace` is server/config-derived for normal clients (not a multi-tenant identity boundary).
- `session_id` must be supplied by the client; there is no shared `"default"` session.
- Not a multi-user SaaS. Do not claim tenant isolation beyond the single-owner model.

## Resume bullets (supported in-repo)

These match working code and tests. Do not claim public cloud traffic, measured chunk-sweep recall, or Langfuse findings until those gates are closed.

- Built a LangGraph agent with conditional routing across retrieval, structured metadata lookup, and a confirmation-gated write action, creating an explicit approval boundary before any store mutation.
- Implemented a pgvector-backed ingestion and retrieval path over Postgres with configurable chunking, embedding, and HNSW index settings exposed through config.
- Built a FastAPI service layer exposing health, readiness, ingestion, invoke, and streaming endpoints, with durable session handling and required API key protection outside local `ENV=dev`.
- Abstracted the model provider so the same agent graph runs against local Ollama or the hosted Anthropic API without changes to the graph, and containerized the service with Docker.
- Added regression tests covering agent write behavior after finding a failure where a disabled store reported an uncommitted write as applied.

## Evidence map

| Bullet | Primary files | Tests |
|--------|---------------|-------|
| LangGraph + gated write | `agent/graph.py`, `agent/tools.py`, `agent/confirmation.py` | `tests/test_agent_routing.py` |
| pgvector path | `vector_store/pgvector_store.py`, `chunking.py`, `embedders.py`, `config.py` | `tests/test_chunking.py`, `tests/test_embedders.py` |
| FastAPI service | `service/main.py` | `tests/test_service_health.py`, `tests/test_service_agent.py` |
| Provider + Docker | `providers/`, `Dockerfile.service`, `docker-compose.yml` | provider unit tests (existing) |
| Write regression | `agent/tools.py` (`retrieval_disabled`) | `tests/test_service_agent.py::test_confirm_on_noop_does_not_claim_applied` |

## Still gated (do not claim yet)

| Claim | Gate |
|-------|------|
| Public cloud URL / live traffic | `fly deploy` + external `GET /readyz` 200 |
| Recall@k across three chunk sizes | run `scripts/chunk_sweep.py` and record numbers |
| Langfuse latency finding | enable keys (with text scrubbing) and read real traces |
| CI blocks merges on live ASR | set `JOURNAL_AGENT_URL` and enforce branch protection |

## Flask lab (not product)

Treat as local-only reference code:

- Crisis / tone / grounding floors historically lived in `app.py`; the service path shares them via `safety/` when wired.
- Session cookie history and `/transcribe` are lab features — do not advertise them as Journal Agent product capabilities.
- Flask Dockerfile is not the supported deploy path; use `Dockerfile.service`.


## Dependency notes

- `chromadb` may still show `pip-audit` advisories without a fixed release; CI reports them with `continue-on-error` until upstream ships a fix.
## Deploy

See [DEPLOY.md](DEPLOY.md) for Fly.io + Neon + Groq (free OSS) / Anthropic (paid).

## Run

```bash
./start.sh --service          # FastAPI on :8080
./start.sh                    # Flask UI on :5000
make service-test             # agent/service regression suite
docker compose up --build     # local Postgres+pgvector + API
```

## Architecture

```
Client → FastAPI (service/main.py)
           ├─ /healthz, /readyz
           ├─ /v1/ingest          → chunk → embed → pgvector
           └─ /v1/agent/*         → LangGraph
                                      ├─ retrieve_journal
                                      ├─ query_entry_metadata
                                      └─ propose_write → confirm_gate → apply
           └─ Langfuse spans (optional)
```

Internal RAG namespace default remains `ai-health-journal` (storage key, not product name).
