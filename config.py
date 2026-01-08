from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Config:
    generator_model: str
    fallback_model: str
    verifier_model: str
    prompt_model: str
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


def load_config() -> Config:
    g = os.getenv
    return Config(
        generator_model=g("GENERATOR_MODEL", "phi3:3.8b"),
        fallback_model=g("FALLBACK_MODEL", "phi3:3.8b"),
        verifier_model=g("VERIFIER_MODEL", "samantha-mistral:7b"),
        prompt_model=g("PROMPT_MODEL", "samantha-mistral:7b"),
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
    )

