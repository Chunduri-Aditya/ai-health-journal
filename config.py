from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Config:
    generator_model: str
    fallback_model: str
    verifier_model: str
    prompt_model: str
    model_selection_strategy: str
    model_machine_tier_override: str
    quality_mode_default: bool
    retrieval_enabled: bool
    retrieval_top_k: int
    groundedness_threshold: float
    vector_backend: str
    privacy_mode: str
    allow_cloud_vectorstore: bool
    whisper_model: str
    whisper_max_audio_bytes: int
    local_cache_max_items: int
    local_cache_ttl_days: int
    local_cache_path: str
    rag_namespace_mode: str
    rag_namespace_fixed: str
    rag_user_id_header: str
    history_personalization_enabled: bool
    # ── LLM backend (Upgrade 08) ─────────────────────────────────────────────
    # llm_backend: "ollama" (default, local) | "anthropic" (cloud, opt-in).
    # The cloud path is double-gated: llm_backend must be "anthropic" AND
    # allow_cloud_llm must be True. Mirrors the ALLOW_CLOUD_VECTORSTORE gate.
    llm_backend: str
    allow_cloud_llm: bool
    # Per-role Anthropic model overrides. Defaults are set to current Sonnet
    # for generator/verifier/fallback (quality) and Haiku for prompt (cost).
    # Override via env to pin specific versions without touching code.
    anthropic_generator_model: str
    anthropic_verifier_model: str
    anthropic_prompt_model: str


def load_config() -> Config:
    g = os.getenv
    return Config(
        generator_model=g("GENERATOR_MODEL", "phi3:3.8b"),
        fallback_model=g("FALLBACK_MODEL", "phi3:3.8b"),
        verifier_model=g("VERIFIER_MODEL", "samantha-mistral:7b"),
        prompt_model=g("PROMPT_MODEL", "samantha-mistral:7b"),
        model_selection_strategy=g("MODEL_SELECTION_STRATEGY", "balanced").lower(),
        model_machine_tier_override=g("MODEL_MACHINE_TIER_OVERRIDE", "auto").lower(),
        quality_mode_default=g("QUALITY_MODE_DEFAULT", "false").lower() == "true",
        retrieval_enabled=g("RETRIEVAL_ENABLED", "false").lower() == "true",
        retrieval_top_k=int(g("RETRIEVAL_TOP_K", "3")),
        groundedness_threshold=float(g("GROUNDEDNESS_THRESHOLD", "0.75")),
        vector_backend=g("VECTOR_BACKEND", "none").lower(),
        privacy_mode=g("PRIVACY_MODE", "balanced").lower(),
        allow_cloud_vectorstore=g("ALLOW_CLOUD_VECTORSTORE", "false").lower() == "true",
        whisper_model=g("WHISPER_MODEL", "base"),
        whisper_max_audio_bytes=int(g("WHISPER_MAX_AUDIO_BYTES", str(15 * 1024 * 1024))),
        local_cache_max_items=int(g("LOCAL_CACHE_MAX_ITEMS", "2000")),
        local_cache_ttl_days=int(g("LOCAL_CACHE_TTL_DAYS", "30")),
        local_cache_path=g("LOCAL_TEXT_CACHE_PATH", "privacy/local_text_cache.jsonl"),
        rag_namespace_mode=g("RAG_NAMESPACE_MODE", "session").lower(),
        rag_namespace_fixed=g("RAG_NAMESPACE_FIXED", "ai-health-journal"),
        rag_user_id_header=g("RAG_USER_ID_HEADER", "X-User-Id"),
        history_personalization_enabled=g("HISTORY_PERSONALIZATION_ENABLED", "false").lower() == "true",
        # Upgrade 08 — LLM backend gate
        llm_backend=g("LLM_BACKEND", "ollama").lower(),
        allow_cloud_llm=g("ALLOW_CLOUD_LLM", "false").lower() == "true",
        anthropic_generator_model=g("ANTHROPIC_GENERATOR_MODEL", "claude-sonnet-4-6"),
        anthropic_verifier_model=g("ANTHROPIC_VERIFIER_MODEL", "claude-sonnet-4-6"),
        anthropic_prompt_model=g("ANTHROPIC_PROMPT_MODEL", "claude-haiku-4-5-20251001"),
    )
