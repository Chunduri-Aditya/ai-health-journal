# GateGuard: callers app.py, service/main.py, agent/graph.py, vector_store/factory.py, tests. User: Implement the plan as specified. Do NOT edit the plan file.
from __future__ import annotations

import logging
import os
from typing import Any, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_LOWERCASE_FIELDS = (
    "model_selection_strategy",
    "model_machine_tier_override",
    "vector_backend",
    "privacy_mode",
    "rag_namespace_mode",
    "embedding_backend",
    "llm_backend",
)

_CLOUD_LLM_BACKENDS = frozenset({"anthropic", "openai_compatible"})


def _hash_embedder_allowed() -> bool:
    if os.getenv("ALLOW_HASH_EMBEDDER", "").lower() == "true":
        return True
    return (os.getenv("ENV") or "dev").strip().lower() in {"test"}


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    generator_model: str = "phi3:3.8b"
    fallback_model: str = "phi3:3.8b"
    verifier_model: str = "samantha-mistral:7b"
    prompt_model: str = "samantha-mistral:7b"
    model_selection_strategy: str = "balanced"
    model_machine_tier_override: str = "auto"
    quality_mode_default: bool = True
    # Block coding/Q&A abuse before Draft→Verify→Revise (saves cloud TPM).
    journal_relevance_gate: bool = True
    retrieval_enabled: bool = False
    retrieval_top_k: int = 3
    groundedness_threshold: float = 0.75
    vector_backend: str = "none"
    privacy_mode: str = "balanced"
    allow_cloud_vectorstore: bool = False
    whisper_model: str = "base"
    whisper_max_audio_bytes: int = 15 * 1024 * 1024
    local_cache_max_items: int = 2000
    local_cache_ttl_days: int = 30
    local_cache_path: str = "privacy/local_text_cache.jsonl"
    rag_namespace_mode: str = "fixed"
    rag_namespace_fixed: str = "ai-health-journal"
    rag_user_id_header: str = "X-User-Id"
    embedding_backend: str = "default"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_url: str = "http://localhost:11434"
    reference_corpus_enabled: bool = False
    reference_top_k: int = 2
    reference_namespace: str = "reference:psychology"
    history_personalization_enabled: bool = False
    llm_backend: str = "ollama"
    allow_cloud_llm: bool = False
    anthropic_generator_model: str = "claude-sonnet-4-6"
    anthropic_verifier_model: str = "claude-sonnet-4-6"
    anthropic_prompt_model: str = "claude-haiku-4-5-20251001"
    # GateGuard: callers factory/roles/tests. Affected API: openai_compatible_* model defaults.
    # Data schemas: Config fields. User: Groq org limits (8K TPM) + model catalog.
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    # Groq free-tier / 8K TPM stack (see docs/DEPLOY.md):
    #   generator = draft on small model (most calls)
    #   fallback  = revise on larger model (only when rewrite needed)
    #   verifier/prompt = small JSON/tool model
    openai_compatible_generator_model: str = "openai/gpt-oss-20b"
    openai_compatible_fallback_model: str = "openai/gpt-oss-120b"
    openai_compatible_verifier_model: str = "openai/gpt-oss-20b"
    openai_compatible_prompt_model: str = "openai/gpt-oss-20b"
    database_url: str = ""
    chunk_size: int = 400
    chunk_overlap: int = 80
    embedding_dimension: int = 0
    openai_embed_model: str = "text-embedding-3-small"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    langfuse_enabled: bool = False
    agent_confirm_writes: bool = True
    trace_include_text: bool = False

    @field_validator(*_LOWERCASE_FIELDS, mode="before")
    @classmethod
    def _lowercase(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def _empty_embedding_dimension(cls, value: Any) -> Any:
        if value in ("", None):
            return 0
        return value

    @model_validator(mode="after")
    def _cross_field_rules(self) -> Self:
        if self.llm_backend in _CLOUD_LLM_BACKENDS:
            # Soft check: factory still falls back; raise only in production ENV.
            env = (os.getenv("ENV") or "dev").strip().lower()
            if not self.allow_cloud_llm:
                msg = (
                    f"llm_backend={self.llm_backend} requires ALLOW_CLOUD_LLM=true "
                    "(allow_cloud_llm must be enabled for the cloud LLM path)."
                )
                if env not in ("dev", "test"):
                    raise ValueError(msg)
                logger.warning("%s Provider factory will fall back to Ollama.", msg)
            elif self.llm_backend == "anthropic" and not os.getenv(
                "ANTHROPIC_API_KEY", ""
            ).strip():
                logger.warning(
                    "llm_backend=anthropic and ALLOW_CLOUD_LLM=true but "
                    "ANTHROPIC_API_KEY is unset; the provider factory will "
                    "fall back to Ollama at runtime."
                )
            elif self.llm_backend == "openai_compatible":
                key = (
                    self.openai_compatible_api_key.strip()
                    or os.getenv("OPENAI_COMPATIBLE_API_KEY", "").strip()
                )
                base = self.openai_compatible_base_url.strip()
                if not key:
                    logger.warning(
                        "llm_backend=openai_compatible and ALLOW_CLOUD_LLM=true but "
                        "OPENAI_COMPATIBLE_API_KEY is unset; the provider factory "
                        "will fall back to Ollama at runtime."
                    )
                if not base:
                    logger.warning(
                        "llm_backend=openai_compatible and ALLOW_CLOUD_LLM=true but "
                        "OPENAI_COMPATIBLE_BASE_URL is unset; the provider factory "
                        "will fall back to Ollama at runtime."
                    )

        if self.vector_backend == "pgvector" and not self.database_url.strip():
            raise ValueError(
                "vector_backend=pgvector requires DATABASE_URL to be set."
            )

        if self.embedding_backend == "hash" and not _hash_embedder_allowed():
            raise ValueError(
                "embedding_backend=hash is only permitted when ENV=test or "
                "ALLOW_HASH_EMBEDDER=true."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )

        return self


def load_config() -> Config:
    """Load and validate application configuration from the environment."""
    return Config()
