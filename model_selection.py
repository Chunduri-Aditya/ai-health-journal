from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Dict, List, Optional

import requests

from config import Config
from system_profile import SystemProfile, detect_system_profile

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_TIMEOUT_SECONDS = 1.5
_CACHE_TTL_SECONDS = 15.0

_installed_cache: Optional[List[str]] = None
_installed_cache_ts = 0.0


@dataclass(frozen=True)
class RuntimeModelSelection:
    generator: str
    fallback: str
    verifier: str
    prompt: str
    strategy: str
    source: str
    summary: str
    preferred: Dict[str, str]
    missing_preferred: List[str]
    available_models: List[str]
    system_profile: SystemProfile

    def to_api_payload(self) -> Dict[str, object]:
        return {
            "generator": self.generator,
            "fallback": self.fallback,
            "verifier": self.verifier,
            "prompt": self.prompt,
            "strategy": self.strategy,
            "source": self.source,
            "summary": self.summary,
            "preferred": dict(self.preferred),
            "missing_preferred": list(self.missing_preferred),
            "available_models": list(self.available_models),
            "system_profile": self.system_profile.to_dict(),
        }


ROLE_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "compact": {
        "generator": ["gemma3:4b", "qwen3:4b", "qwen2.5:3b", "phi3:3.8b"],
        "fallback": ["gemma3:4b", "qwen3:4b", "qwen2.5:3b", "phi3:3.8b"],
        "verifier": ["qwen3:4b", "gemma3:4b", "qwen2.5:3b", "phi3:3.8b", "samantha-mistral:7b"],
        "prompt": ["gemma3:4b", "qwen3:4b", "qwen2.5:3b", "phi3:3.8b", "samantha-mistral:7b"],
    },
    "balanced": {
        "generator": ["qwen3:8b", "gemma3:12b", "qwen2.5:7b", "gemma3:4b", "phi3:3.8b"],
        "fallback": ["qwen3:8b", "gemma3:12b", "qwen2.5:7b", "gemma3:4b", "phi3:3.8b"],
        "verifier": ["qwen3:8b", "qwen2.5:7b", "gemma3:4b", "phi3:3.8b", "samantha-mistral:7b"],
        "prompt": ["gemma3:4b", "qwen3:4b", "qwen2.5:3b", "phi3:3.8b", "samantha-mistral:7b"],
    },
    "high-memory": {
        "generator": ["mistral-small3.2:24b", "qwen3:14b", "qwen2.5:14b", "qwen3:8b", "gemma3:12b", "phi3:3.8b"],
        "fallback": ["mistral-small3.2:24b", "qwen3:14b", "qwen2.5:14b", "qwen3:8b", "gemma3:12b", "phi3:3.8b"],
        "verifier": ["deepseek-r1:8b", "qwen3:8b", "qwen2.5:7b", "gemma3:4b", "samantha-mistral:7b"],
        "prompt": ["gemma3:4b", "qwen3:4b", "qwen2.5:3b", "samantha-mistral:7b"],
    },
}


def _env_override(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalized(name: str) -> str:
    return name.strip().lower()


def _is_candidate_chat_model(name: str) -> bool:
    normalized = _normalized(name)
    blocked_tokens = (
        "embed", "embedding", "bge-", "nomic-embed", "all-minilm",
        # Code-generation models are poor fits for emotionally-aware
        # journaling and were previously selectable in the model dropdown
        # alongside real candidates like phi3/qwen3.
        "codellama", "coder", "starcoder", "codegemma",
    )
    return not any(token in normalized for token in blocked_tokens)


def discover_installed_ollama_models(force_refresh: bool = False) -> List[str]:
    global _installed_cache, _installed_cache_ts
    now = time.time()
    if not force_refresh and _installed_cache is not None and (now - _installed_cache_ts) < _CACHE_TTL_SECONDS:
        return list(_installed_cache)

    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        names = sorted(
            {
                model.get("name", "").strip()
                for model in models
                if isinstance(model, dict) and model.get("name") and _is_candidate_chat_model(model["name"])
            }
        )
    except Exception:
        names = []

    _installed_cache = names
    _installed_cache_ts = now
    return list(names)


def _find_installed(preferred: List[str], installed_names: List[str]) -> Optional[str]:
    if not installed_names:
        return None

    normalized_map = {_normalized(name): name for name in installed_names}
    for candidate in preferred:
        direct = normalized_map.get(_normalized(candidate))
        if direct:
            return direct

    for candidate in preferred:
        family = candidate.split(":", 1)[0].lower()
        tag = candidate.split(":", 1)[1].lower() if ":" in candidate else ""
        for installed in installed_names:
            installed_norm = _normalized(installed)
            if installed_norm == family:
                return installed
            if installed_norm.startswith(f"{family}:") and tag and installed_norm.startswith(f"{family}:{tag}"):
                return installed

    return None


def _preferred_models_for_tier(machine_tier: str) -> Dict[str, str]:
    candidates = ROLE_CANDIDATES.get(machine_tier, ROLE_CANDIDATES["balanced"])
    return {role: values[0] for role, values in candidates.items()}


def _pick_active_model(
    role: str,
    cfg_value: str,
    env_name: str,
    strategy: str,
    machine_tier: str,
    installed_names: List[str],
) -> str:
    explicit = _env_override(env_name)
    if explicit:
        return explicit

    if strategy != "manual":
        recommended = _find_installed(ROLE_CANDIDATES.get(machine_tier, ROLE_CANDIDATES["balanced"])[role], installed_names)
        if recommended:
            return recommended

    config_installed = _find_installed([cfg_value], installed_names)
    if config_installed:
        return config_installed

    if installed_names:
        return installed_names[0]

    return cfg_value


def get_runtime_model_selection(cfg: Config) -> RuntimeModelSelection:
    strategy = (cfg.model_selection_strategy or "balanced").strip().lower()
    profile = detect_system_profile(cfg.model_machine_tier_override)
    machine_tier = profile.machine_tier
    preferred = _preferred_models_for_tier(machine_tier)
    installed_names = discover_installed_ollama_models()

    generator = _pick_active_model("generator", cfg.generator_model, "GENERATOR_MODEL", strategy, machine_tier, installed_names)
    fallback = _pick_active_model("fallback", cfg.fallback_model, "FALLBACK_MODEL", strategy, machine_tier, installed_names)
    verifier = _pick_active_model("verifier", cfg.verifier_model, "VERIFIER_MODEL", strategy, machine_tier, installed_names)
    prompt = _pick_active_model("prompt", cfg.prompt_model, "PROMPT_MODEL", strategy, machine_tier, installed_names)

    missing_preferred = list(
        dict.fromkeys(
            model_name
            for model_name in preferred.values()
            if installed_names and model_name not in installed_names
        )
    )

    if strategy == "manual":
        source = "manual"
        summary = "Manual model selection is active — runtime models come from explicit configuration."
    elif installed_names:
        all_preferred_installed = all(name in installed_names for name in preferred.values())
        if all_preferred_installed:
            source = "balanced-installed"
            summary = f"Balanced preset matched this {machine_tier} machine — all recommended models are installed."
        else:
            source = "balanced-fallback"
            summary = (
                f"Balanced preset targeted this {machine_tier} machine, but some recommended models "
                f"aren't installed, so it's using the best local fit."
            )
    else:
        source = "configured-defaults"
        summary = (
            f"Balanced preset detected a {machine_tier} machine, but the Ollama model inventory was "
            f"unavailable, so the app is using configured defaults."
        )

    return RuntimeModelSelection(
        generator=generator,
        fallback=fallback,
        verifier=verifier,
        prompt=prompt,
        strategy=strategy,
        source=source,
        summary=summary,
        preferred=preferred,
        missing_preferred=missing_preferred,
        available_models=installed_names,
        system_profile=profile,
    )
