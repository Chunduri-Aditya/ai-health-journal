# GateGuard: callers agent/respond.py, agent/intent.py, app.py, tests.
# Affected API: resolve_role_models(cfg) -> RoleModels; is_cloud_llm_active(cfg).
# Data schemas: RoleModels dataclass (generator, verifier, prompt, fallback).
# User: Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.
"""Resolve generator / verifier / prompt / fallback model IDs per LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoleModels:
    generator: str
    verifier: str
    prompt: str
    fallback: str


_CLOUD_BACKENDS = frozenset({"anthropic", "openai_compatible"})


def resolve_role_models(cfg: Any) -> RoleModels:
    """Return role model names for the active backend.

    Cloud backends require ``allow_cloud_llm`` so a mis-set ``LLM_BACKEND``
    without the privacy gate still uses local Ollama model names.
    """
    backend = (getattr(cfg, "llm_backend", "") or "").lower()
    gate = bool(getattr(cfg, "allow_cloud_llm", False))

    if backend == "anthropic" and gate:
        gen = getattr(cfg, "anthropic_generator_model", "claude-sonnet-4-6")
        return RoleModels(
            generator=gen,
            verifier=getattr(cfg, "anthropic_verifier_model", gen),
            prompt=getattr(cfg, "anthropic_prompt_model", "claude-haiku-4-5-20251001"),
            fallback=gen,
        )

    if backend == "openai_compatible" and gate:
        # GateGuard: callers app/agent/tests. Affected API: RoleModels for Groq.
        # Data schemas: RoleModels. User: Groq org limits (8K TPM chat).
        # Draft/verify/prompts on 20B; escalate revise to 120B only when needed.
        gen = getattr(
            cfg, "openai_compatible_generator_model", "openai/gpt-oss-20b"
        )
        return RoleModels(
            generator=gen,
            verifier=getattr(
                cfg, "openai_compatible_verifier_model", "openai/gpt-oss-20b"
            ),
            prompt=getattr(
                cfg, "openai_compatible_prompt_model", "openai/gpt-oss-20b"
            ),
            fallback=getattr(
                cfg, "openai_compatible_fallback_model", "openai/gpt-oss-120b"
            )
            or gen,
        )

    return RoleModels(
        generator=getattr(cfg, "generator_model", "phi3:3.8b"),
        verifier=getattr(cfg, "verifier_model", "samantha-mistral:7b"),
        prompt=getattr(cfg, "prompt_model", "samantha-mistral:7b"),
        fallback=getattr(cfg, "fallback_model", "phi3:3.8b"),
    )


def is_cloud_llm_active(cfg: Any) -> bool:
    backend = (getattr(cfg, "llm_backend", "") or "").lower()
    return bool(getattr(cfg, "allow_cloud_llm", False)) and backend in _CLOUD_BACKENDS
