# Deploy Journal Agent (Fly.io + cloud LLMs)

Privacy note: with cloud LLM/embeddings enabled, journal text leaves your machine.
Keep `PRIVACY_MODE=balanced` or `strict`, and do not use real clinical notes while testing.

This runbook deploys the **FastAPI** product (`Dockerfile.service`). The Flask UI is local-only.

## Stack

| Layer | Free OSS test | Paid flip |
|-------|---------------|-----------|
| App host | [Fly.io](https://fly.io) | same |
| Chat LLM | Groq (`LLM_BACKEND=openai_compatible`) | Anthropic (`LLM_BACKEND=anthropic`) |
| Embeddings | OpenAI `text-embedding-3-small` | same |
| Vector DB | Neon Postgres + pgvector | same |
| Auth | `JOURNAL_AGENT_API_KEY` via `X-API-Key` | same |

`fly.toml` defaults to the **Groq free-OSS** chat profile with `ENV=production`.

## 1. Accounts and keys

1. **Neon** — [console.neon.tech](https://console.neon.tech): create a project, enable the `vector` extension (`CREATE EXTENSION IF NOT EXISTS vector;`), copy the connection string (`sslmode=require`).
2. **Groq** — [console.groq.com](https://console.groq.com): create an API key (free tier, rate-limited).
3. **OpenAI** — [platform.openai.com](https://platform.openai.com): API key for embeddings (prepaid / trial credits).
4. **Fly** — `fly auth login` then `fly apps create journal-agent` (or rename `app` in `fly.toml`).

Optional later: Anthropic Console key for the paid chat flip.

## 2. Local smoke (no Fly)

```bash
export ENV=dev
export LLM_BACKEND=openai_compatible
export ALLOW_CLOUD_LLM=true
export OPENAI_COMPATIBLE_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_COMPATIBLE_API_KEY=gsk-...
# GateGuard: callers operators. Affected API: local Groq smoke env.
# Data schemas: none. User: Groq org limits (8K TPM).
export OPENAI_COMPATIBLE_GENERATOR_MODEL=openai/gpt-oss-20b
export OPENAI_COMPATIBLE_FALLBACK_MODEL=openai/gpt-oss-120b
export OPENAI_COMPATIBLE_VERIFIER_MODEL=openai/gpt-oss-20b
export OPENAI_COMPATIBLE_PROMPT_MODEL=openai/gpt-oss-20b

# Under ~8K TPM: draft+verify on 20B; 120B only for revise when rewrite_required.
# Still hitting limits: set FALLBACK to openai/gpt-oss-20b (or qwen/qwen3.6-27b).
# Note: HTTP 401 is an invalid API key — not a rate-limit error (429).

# Chat-only smoke (skip RAG):
export RETRIEVAL_ENABLED=false
export VECTOR_BACKEND=none

./start.sh --service
curl -sS http://127.0.0.1:8080/readyz
curl -sS -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","message":"I slept well after a walk."}' \
  http://127.0.0.1:8080/v1/agent/invoke
```

Full RAG locally against Neon:

```bash
export RETRIEVAL_ENABLED=true
export VECTOR_BACKEND=pgvector
export ALLOW_CLOUD_VECTORSTORE=true
export DATABASE_URL='postgresql://USER:PASS@HOST/dbname?sslmode=require'
export EMBEDDING_BACKEND=openai
export OPENAI_API_KEY=sk-...
export EMBEDDING_DIMENSION=1536
```

## 3. Fly secrets + deploy

```bash
fly secrets set \
  JOURNAL_AGENT_API_KEY="$(openssl rand -hex 32)" \
  DATABASE_URL='postgresql://USER:PASS@HOST/dbname?sslmode=require' \
  ALLOW_CLOUD_VECTORSTORE=true \
  ALLOW_CLOUD_LLM=true \
  OPENAI_COMPATIBLE_API_KEY=gsk-... \
  OPENAI_API_KEY=sk-...

fly deploy
```

Non-secret defaults live in `fly.toml` (`ENV=production`, Groq base URL, embedding model, pgvector).

## 4. Verify

```bash
APP=https://journal-agent.fly.dev   # your app hostname
curl -sS "$APP/readyz"
curl -sS -H "X-API-Key: $JOURNAL_AGENT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","message":"I slept well after a walk."}' \
  "$APP/v1/agent/invoke"
```

Expect `/readyz` HTTP 200 with healthy store + LLM. Missing `JOURNAL_AGENT_API_KEY` in production returns **503** on `/v1/*`; wrong key returns **401**.

## 5. Flip to Anthropic (paid)

Keep Neon + OpenAI embeddings. Change chat only:

```bash
# Option A — secrets override (leave fly.toml as-is for next free-OSS redeploy):
fly secrets set LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-...

# Option B — edit fly.toml [env] LLM_BACKEND = "anthropic", then fly deploy
```

Role models default to `claude-sonnet-4-6` (generator/verifier) and Haiku for prompt classification (`ANTHROPIC_*_MODEL` env overrides).

To return to Groq: `fly secrets unset ANTHROPIC_API_KEY` (if set) and `fly secrets set LLM_BACKEND=openai_compatible` or restore `fly.toml` and redeploy.

## 6. Checklist

- [ ] Neon `vector` extension enabled
- [ ] `ENV=production` (set in `fly.toml`)
- [ ] `JOURNAL_AGENT_API_KEY` set as a Fly secret
- [ ] `ALLOW_CLOUD_LLM=true` and `ALLOW_CLOUD_VECTORSTORE=true`
- [ ] Groq key **or** Anthropic key present for the chosen `LLM_BACKEND`
- [ ] `OPENAI_API_KEY` present for embeddings
- [ ] `curl /readyz` → 200
- [ ] Authenticated `POST /v1/agent/invoke` returns a response

## Related

- [JOURNAL_AGENT.md](JOURNAL_AGENT.md) — product surface and evidence map
- [PRIVACY.md](../PRIVACY.md) — cloud gates
- Upgrade 08 — Anthropic provider design
