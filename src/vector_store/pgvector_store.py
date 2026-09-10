"""pgvector-backed VectorStore for managed Postgres.

Ingestion chunks journal entries (configurable size/overlap), embeds each
chunk, and upserts into a `journal_chunks` table with an HNSW index.
Query uses cosine distance (`<=>`) scoped by namespace.

Requires:
  - DATABASE_URL pointing at Postgres with the `vector` extension
  - `psycopg[binary]`, `psycopg_pool`, and `pgvector` Python packages
  - ALLOW_CLOUD_VECTORSTORE=true when the DB is not localhost (same gate as Pinecone)

GateGuard facts: callers vector_store/factory.py, scripts/chunk_sweep.py,
agent/tools.py. Existing file. Schema: journal_chunks(id, entry_id, namespace,
chunk_index, text, metadata jsonb, embedding vector(dim), created_at) and
agent_sessions(session_id, payload jsonb, updated_at). User: Implement the plan
as specified, it is attached for your reference. Do NOT edit the plan file itself.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import RetrievalHit, VectorStore
from .chunking import ChunkingConfig, chunk_text
from .embedders import Embedder, HashEmbedder

logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = "default"


def _is_local_database_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "postgres", "db"}


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised when the embedder dimension does not match the table column."""


class PgVectorStore(VectorStore):
    """Managed-Postgres + pgvector implementation of VectorStore."""

    def __init__(
        self,
        *,
        database_url: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        chunking: Optional[ChunkingConfig] = None,
        default_namespace: str = _DEFAULT_NAMESPACE,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 40,
        allow_cloud: bool = False,
        pool_min: int = 1,
        pool_max: int = 10,
    ) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            from psycopg_pool import ConnectionPool
        except ImportError as e:
            raise RuntimeError(
                "pgvector backend selected but psycopg/pgvector/psycopg_pool are not installed.\n"
                "Install: pip install -r requirements-service.txt"
            ) from e

        self._psycopg = psycopg
        self._register_vector = register_vector

        url = database_url or os.getenv("DATABASE_URL", "")
        if not url:
            raise RuntimeError(
                "pgvector backend selected but DATABASE_URL is not set."
            )
        if not allow_cloud and not _is_local_database_url(url):
            raise RuntimeError(
                "cloud_vectorstore_not_enabled: DATABASE_URL points at a remote host. "
                "Set ALLOW_CLOUD_VECTORSTORE=true to enable managed Postgres/pgvector."
            )

        self._database_url = url
        self._embedder = embedder or HashEmbedder()
        self._chunking = chunking or ChunkingConfig()
        self._default_namespace = default_namespace or _DEFAULT_NAMESPACE
        self._hnsw_m = max(2, int(hnsw_m))
        self._hnsw_ef_construction = max(4, int(hnsw_ef_construction))
        self._hnsw_ef_search = max(1, int(hnsw_ef_search))
        self._dimension = self._embedder.dimension
        self._dimension_ok = True
        self._dimension_error: Optional[str] = None

        def _configure(conn):
            self._register_vector(conn)

        self._pool = ConnectionPool(
            conninfo=url,
            min_size=max(1, int(pool_min)),
            max_size=max(1, int(pool_max)),
            kwargs={"autocommit": True},
            configure=_configure,
            open=True,
        )

        self._ensure_schema()
        self._probe_dimension()
        logger.info(
            "PgVectorStore ready (dim=%s, chunk_size=%s, overlap=%s, hnsw_m=%s, pool=%s-%s).",
            self._dimension,
            self._chunking.chunk_size,
            self._chunking.chunk_overlap,
            self._hnsw_m,
            pool_min,
            pool_max,
        )

    @property
    def enabled(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "pgvector"

    @property
    def dimension_ok(self) -> bool:
        return self._dimension_ok

    @property
    def dimension_error(self) -> Optional[str]:
        return self._dimension_error

    def close(self) -> None:
        try:
            self._pool.close()
        except Exception:
            pass

    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        return namespace or self._default_namespace

    def _ensure_schema(self) -> None:
        dim = self._dimension
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS journal_chunks (
                        id TEXT PRIMARY KEY,
                        entry_id TEXT NOT NULL,
                        namespace TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dim}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        session_id TEXT PRIMARY KEY,
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS journal_chunks_namespace_idx
                    ON journal_chunks (namespace)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS journal_chunks_entry_id_idx
                    ON journal_chunks (entry_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS journal_chunks_embedding_hnsw
                    ON journal_chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef_construction})
                    """
                )

    def _probe_dimension(self) -> None:
        """Compare embedder dimension to the existing vector column atttypmod."""
        try:
            probe = self._embedder.embed_query("journal-agent-dim-probe")
            measured = len(probe)
            if measured != self._dimension:
                self._dimension = measured
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT a.atttypmod
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE c.relname = 'journal_chunks'
                          AND a.attname = 'embedding'
                          AND n.nspname = current_schema()
                        """
                    )
                    row = cur.fetchone()
            if not row or row[0] is None:
                self._dimension_ok = True
                return
            # pgvector stores dimension in atttypmod (value == dim for vector(n)).
            col_dim = int(row[0])
            if col_dim > 0 and col_dim != self._dimension:
                self._dimension_ok = False
                self._dimension_error = (
                    f"embedder dimension {self._dimension} != table vector({col_dim}). "
                    "Re-ingest after aligning EMBEDDING_BACKEND / EMBEDDING_DIMENSION, "
                    "or drop journal_chunks and recreate."
                )
                logger.error(self._dimension_error)
            else:
                self._dimension_ok = True
                self._dimension_error = None
        except Exception as e:
            self._dimension_ok = False
            self._dimension_error = f"dimension probe failed: {e}"
            logger.error(self._dimension_error)

    @staticmethod
    def _chunk_id(entry_id: str, chunk_index: int) -> str:
        return f"{entry_id}::chunk::{chunk_index}"

    def add_entry(
        self,
        entry_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        namespace: Optional[str] = None,
    ) -> bool:
        if not self._dimension_ok:
            logger.error("PgVectorStore.add_entry blocked: %s", self._dimension_error)
            return False
        ns = self._resolve_namespace(namespace)
        chunks = chunk_text(text, self._chunking)
        if not chunks:
            return False

        try:
            vectors = self._embedder.embed_documents([c.text for c in chunks])
            base_meta = dict(metadata or {})
            base_meta.setdefault("namespace", ns)
            rows = []
            for chunk, vec in zip(chunks, vectors):
                meta = dict(base_meta)
                meta["chunk_index"] = chunk.index
                meta["start_char"] = chunk.start_char
                meta["end_char"] = chunk.end_char
                rows.append(
                    (
                        self._chunk_id(entry_id, chunk.index),
                        entry_id,
                        ns,
                        chunk.index,
                        chunk.text,
                        json.dumps(meta),
                        vec,
                    )
                )

            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM journal_chunks WHERE entry_id = %s AND namespace = %s",
                            (entry_id, ns),
                        )
                        cur.executemany(
                            """
                            INSERT INTO journal_chunks
                                (id, entry_id, namespace, chunk_index, text, metadata, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                text = EXCLUDED.text,
                                metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding,
                                chunk_index = EXCLUDED.chunk_index
                            """,
                            rows,
                        )
            return True
        except Exception as e:
            logger.error("PgVectorStore.add_entry failed: %s", e)
            return False

    def query(
        self,
        text: str,
        *,
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        if top_k <= 0:
            return []
        if not self._dimension_ok:
            logger.error("PgVectorStore.query blocked: %s", self._dimension_error)
            return []
        ns = self._resolve_namespace(namespace)
        try:
            qvec = self._embedder.embed_query(text)
            fetch_n = max(top_k * 4, top_k)
            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "SET LOCAL hnsw.ef_search = %s",
                            (self._hnsw_ef_search,),
                        )
                        sql = """
                            SELECT entry_id, text, metadata,
                                   1 - (embedding <=> %s::vector) AS score
                            FROM journal_chunks
                            WHERE namespace = %s
                        """
                        params: List[Any] = [qvec, ns]
                        if filter_metadata:
                            for key, value in filter_metadata.items():
                                sql += " AND metadata ->> %s = %s"
                                params.extend([str(key), str(value)])
                        sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s"
                        params.extend([qvec, fetch_n])
                        cur.execute(sql, params)
                        rows = cur.fetchall()
        except Exception as e:
            logger.error("PgVectorStore.query failed: %s", e)
            return []

        hits: List[RetrievalHit] = []
        seen_entries: set[str] = set()
        for entry_id, doc, meta, score in rows:
            if entry_id in seen_entries:
                continue
            seen_entries.add(entry_id)
            md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            hits.append(
                RetrievalHit(
                    id=str(entry_id),
                    text=doc or "",
                    score=float(score or 0.0),
                    metadata=dict(md),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def delete_entry(
        self,
        entry_id: str,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        ns = self._resolve_namespace(namespace)
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM journal_chunks WHERE entry_id = %s AND namespace = %s",
                        (entry_id, ns),
                    )
        except Exception as e:
            logger.error("PgVectorStore.delete_entry failed: %s", e)

    def clear_namespace(self, namespace: str) -> None:
        ns = namespace or self._default_namespace
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM journal_chunks WHERE namespace = %s",
                        (ns,),
                    )
        except Exception as e:
            logger.debug("PgVectorStore.clear_namespace(%s) no-op: %s", ns, e)

    def healthcheck(self) -> bool:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return self._dimension_ok
        except Exception as e:
            logger.warning("PgVectorStore healthcheck failed: %s", e)
            return False

    def query_entry_metadata(
        self,
        *,
        namespace: Optional[str] = None,
        entry_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Structured metadata lookup (agent tool surface, not ANN search)."""
        ns = self._resolve_namespace(namespace)
        limit = max(1, min(int(limit), 200))
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    if entry_id:
                        cur.execute(
                            """
                            SELECT DISTINCT ON (entry_id)
                                entry_id, metadata, created_at,
                                LEFT(text, 200) AS preview
                            FROM journal_chunks
                            WHERE namespace = %s AND entry_id = %s
                            ORDER BY entry_id, chunk_index
                            """,
                            (ns, entry_id),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT DISTINCT ON (entry_id)
                                entry_id, metadata, created_at,
                                LEFT(text, 200) AS preview
                            FROM journal_chunks
                            WHERE namespace = %s
                            ORDER BY entry_id, chunk_index
                            LIMIT %s
                            """,
                            (ns, limit),
                        )
                    rows = cur.fetchall()
        except Exception as e:
            logger.error("PgVectorStore.query_entry_metadata failed: %s", e)
            return []

        out: List[Dict[str, Any]] = []
        for eid, meta, created_at, preview in rows:
            md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            out.append(
                {
                    "entry_id": eid,
                    "metadata": md,
                    "created_at": created_at.isoformat() if created_at else None,
                    "preview": preview,
                }
            )
        return out

    # ── Durable agent sessions (HITL confirm across process restarts) ─────────
    def load_session(self, session_id: str) -> Dict[str, Any]:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload FROM agent_sessions WHERE session_id = %s",
                        (session_id,),
                    )
                    row = cur.fetchone()
            if not row:
                return {}
            payload = row[0]
            return dict(payload) if isinstance(payload, dict) else json.loads(payload or "{}")
        except Exception as e:
            logger.warning("load_session failed: %s", e)
            return {}

    def save_session(self, session_id: str, payload: Dict[str, Any]) -> None:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO agent_sessions (session_id, payload, updated_at)
                        VALUES (%s, %s::jsonb, NOW())
                        ON CONFLICT (session_id) DO UPDATE SET
                            payload = EXCLUDED.payload,
                            updated_at = NOW()
                        """,
                        (session_id, json.dumps(payload)),
                    )
        except Exception as e:
            logger.warning("save_session failed: %s", e)

    def delete_session(self, session_id: str) -> None:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM agent_sessions WHERE session_id = %s",
                        (session_id,),
                    )
        except Exception as e:
            logger.debug("delete_session failed: %s", e)

    def purge_stale_sessions(self, ttl_sec: int) -> None:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM agent_sessions
                        WHERE updated_at < NOW() - (%s || ' seconds')::interval
                        """,
                        (int(ttl_sec),),
                    )
        except Exception as e:
            logger.debug("purge_stale_sessions failed: %s", e)
