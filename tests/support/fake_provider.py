"""Deterministic, theme-aware LLM double for offline testing.

`ThemeAwareFakeProvider` implements the `providers.base.LLMProvider` interface so
it can stand in for `app._provider` and drive the real `/analyze` route without a
live Ollama. Output is a pure function of the entry text: a keyword map picks
emotions/patterns, and quotes are echoed as exact substrings of the entry so they
survive `app._strip_ungrounded_quotes`.

The same logic is exposed as a module-level `fake_json_generate` matching
`llm_client.json_generate`'s signature, so `evals/run_evals.py --mock_llm` can
patch the LLM-as-judge metric calls too. `install_mock_llm()` wires both.

Crisis handling is intentionally *not* relied on here: `app._CRISIS_PATTERNS`
fires on the raw entry regardless of what this double returns, so crisis-turn
assertions hold even if the fake verdict is oblivious. The fake still sets a
safety flag when it sees crisis phrasing, so the judge path sees it too.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from providers.base import LLMProvider

# ── Theme map ──────────────────────────────────────────────────────────────
# Ordered so the first matching theme wins; each theme is (keywords, emotions,
# patterns, coping_suggestions). Deliberately small and disjoint so retrieval
# tests can assert that a query pulls the *matching* prior theme.
_THEMES: List[Tuple[Tuple[str, ...], List[str], List[str], List[str]]] = [
    (
        ("work", "manager", "boss", "deadline", "office", "burnout", "overtime", "project", "workload"),
        ["stress", "overwhelm", "exhaustion"],
        ["overcommitment at work"],
        ["Block one recovery hour that work cannot touch.", "Name the single most urgent task and start only that."],
    ),
    (
        ("sleep", "awake", "insomnia", "restless", "3am", "night", "tired", "exhausted", "foggy"),
        ["fatigue", "frustration"],
        ["disrupted sleep"],
        ["Try a fixed wind-down time tonight.", "Keep the phone out of reach after lights-out."],
    ),
    (
        ("partner", "argue", "argument", "fight", "spouse", "wife", "husband", "relationship", "heard"),
        ["hurt", "sadness", "frustration"],
        ["conflict avoidance in the relationship"],
        ["Write what you needed to hear before raising it again.", "Ask for one specific change, not everything at once."],
    ),
    (
        ("application", "applications", "interview", "rejection", "laid off", "unemployed", "resume", "linkedin", "job hunt", "job market"),
        ["discouragement", "anxiety", "hope"],
        ["job-search fatigue"],
        ["Track effort, not outcomes, for one week.", "Send one message to a person, not a portal."],
    ),
    (
        ("grateful", "thankful", "gratitude", "appreciate", "good things", "lighter"),
        ["contentment", "gratitude"],
        ["noticing what is going well"],
        ["Keep the three-good-things note going.", "Revisit it on a hard day."],
    ),
    (
        ("run", "running", "gym", "exercise", "workout", "walk", "energy"),
        ["motivation", "calm"],
        ["movement and physical energy"],
        ["Anchor the run to a fixed cue each morning.", "Count showing up, not distance."],
    ),
]

_DEFAULT_EMOTIONS = ["reflective"]
_DEFAULT_PATTERNS = ["self-reflection"]
_DEFAULT_COPING = ["Name the feeling in one sentence.", "Notice what it might be pointing at."]

# Reflexive crisis phrasing — a subset of app._CRISIS_PATTERNS. Only used so the
# fake verdict can carry a safety flag; the app's own floor is the real gate.
_CRISIS_HINT = re.compile(
    r"\b(kill(?:ing)?\s+myself|end\s+my\s+life|want\s+to\s+die|hurt\s+myself|"
    r"harm\s+myself|suicid|better\s+off\s+.*dead|disappear)",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# The draft/verifier prompt templates embed the current entry under one of these
# markers, followed by RETRIEVED_CONTEXT or the end of the section. The fake must
# key off the *current entry* only — scanning the whole prompt would let a prior
# turn's retrieved text (e.g. an earlier crisis entry) contaminate this turn's
# theme/quote/crisis decision.
_ENTRY_MARKERS = ("JOURNAL ENTRY:", "CURRENT_ENTRY:")
_ENTRY_END_RE = re.compile(r"\n\s*(RETRIEVED_CONTEXT|Return ONLY|You may reference)", re.IGNORECASE)


def _extract_entry(prompt: str) -> str:
    """Pull the current journal entry out of a draft/verifier prompt.

    Falls back to the whole string if no marker is present (e.g. a caller that
    passes the raw entry directly), so the fake degrades gracefully.
    """
    text = prompt or ""
    for marker in _ENTRY_MARKERS:
        idx = text.find(marker)
        if idx == -1:
            continue
        rest = text[idx + len(marker):]
        end = _ENTRY_END_RE.search(rest)
        return (rest[: end.start()] if end else rest).strip()
    return text.strip()


def _themed(entry: str) -> Tuple[List[str], List[str], List[str]]:
    text = (entry or "").lower()
    for keywords, emotions, patterns, coping in _THEMES:
        if any(k in text for k in keywords):
            return list(emotions), list(patterns), list(coping)
    return list(_DEFAULT_EMOTIONS), list(_DEFAULT_PATTERNS), list(_DEFAULT_COPING)


def _grounded_quote(entry: str) -> List[str]:
    """First sentence (or a short prefix) echoed as an exact substring of entry."""
    entry = entry or ""
    candidate = entry.split(".")[0].strip()
    if candidate and candidate in entry:
        return [candidate[:120]] if candidate[:120] in entry else [candidate]
    return []


def build_analysis(entry: str) -> Dict[str, Any]:
    """A schema-valid AnalysisOutput dict derived purely from the entry."""
    emotions, patterns, coping = _themed(entry)
    # Non-empty reframe for negative themes so the crisis gate's clearing of it
    # is observable; gratitude/health stay neutral (no reframe needed).
    positive = any(w in (entry or "").lower() for w in ("grateful", "thankful", "gratitude", "energy", "calm"))
    reframe = "" if positive else "This is one hard moment, not a verdict on the whole of you."
    return {
        "summary": f"You are processing {patterns[0]}." if patterns else "You are reflecting on the day.",
        "emotions": emotions,
        "patterns": patterns,
        "triggers": [],
        "coping_suggestions": coping,
        "quotes_from_user": _grounded_quote(entry),
        "confidence": 0.7,
        "journaling_feedback": ["Naming the feeling plainly, as you did, is the useful move."],
        "reframe": reframe,
    }


def build_verdict(entry: str) -> Dict[str, Any]:
    """A schema-valid VerifierVerdict dict. High groundedness, no rewrite, so the
    quality pipeline returns the draft deterministically (no fallback hop)."""
    crisis = bool(_CRISIS_HINT.search(entry or ""))
    return {
        "groundedness_score": 0.95,
        "unsupported_claims": [],
        "safety_flags": ["self harm"] if crisis else [],
        "crisis_detected": crisis,
        "rewrite_required": False,
        "rewrite_instructions": "",
    }


def _is_verifier_call(system_prompt: str, validator_model: Optional[type]) -> bool:
    if validator_model is not None and getattr(validator_model, "__name__", "") == "VerifierVerdict":
        return True
    return "verifier" in (system_prompt or "").lower()


class ThemeAwareFakeProvider(LLMProvider):
    """Drop-in for `app._provider`. Deterministic; needs no network."""

    def generate(self, model, prompt, *, system=None, temperature=None, timeout=30) -> str:
        # Used by /prompt and the legacy path. A short, stable string is enough.
        return "What did today ask of you that you did not expect?"

    def json_generate(
        self,
        model,
        system_prompt,
        user_prompt,
        *,
        json_schema=None,
        max_retries=5,
        temperature=None,
        validator_model=None,
    ) -> Dict[str, Any]:
        entry = _extract_entry(user_prompt)
        if _is_verifier_call(system_prompt, validator_model):
            return build_verdict(entry)
        return build_analysis(entry)

    def healthcheck(self) -> bool:
        return True


def fake_json_generate(
    model,
    system_prompt,
    user_prompt,
    max_retries=5,
    json_schema=None,
    temperature=None,
    *,
    validator_model=None,
    return_model=False,
) -> Dict[str, Any]:
    """Matches `llm_client.json_generate` so it can replace the judge-metric path
    in `evals/run_evals.py`. `return_model` is ignored (callers there take dicts)."""
    entry = _extract_entry(user_prompt)
    if _is_verifier_call(system_prompt, validator_model):
        return build_verdict(entry)
    return build_analysis(entry)


def install_mock_llm() -> None:
    """Patch both LLM surfaces for a fully offline eval/smoke run.

    - `app._provider` -> ThemeAwareFakeProvider (the /analyze route's generation)
    - `evals.run_evals.json_generate` -> fake_json_generate (LLM-as-judge metrics)

    Idempotent and import-order tolerant: each patch is skipped if its module
    isn't imported yet in the current process.
    """
    import sys

    app_mod = sys.modules.get("app")
    if app_mod is not None:
        app_mod._provider = ThemeAwareFakeProvider()  # type: ignore[attr-defined]

    run_evals = sys.modules.get("evals.run_evals") or sys.modules.get("run_evals")
    if run_evals is not None:
        run_evals.json_generate = fake_json_generate  # type: ignore[attr-defined]
