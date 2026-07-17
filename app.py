# === app.py ===
# AI Health Journal — Flask backend
# Multi-model quality pipeline with RAG, session tracking, transcription support.

from __future__ import annotations

import html
import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

# ── Config & env ──────────────────────────────────────────────────────────────
load_dotenv()

from config import load_config
from model_selection import get_runtime_model_selection

cfg = load_config()

# ── Quality pipeline modules (graceful degradation if missing) ────────────────
try:
    from llm_client import (
        DRAFT_JSON_SCHEMA,
        VERIFIER_JSON_SCHEMA,
    )
    from generator_prompts import DRAFT_SYSTEM_PROMPT, get_draft_prompt
    from verifier_prompts import (
        VERIFIER_SYSTEM_PROMPT,
        get_revision_prompt,
        get_verifier_prompt,
    )
    from schemas.analysis import AnalysisOutput
    from schemas.verifier import VerifierVerdict
    from vector_store.base import RetrievalHit, format_hits_as_context
    from vector_store.factory import get_vector_store
except ImportError as e:
    logging.warning(f"Quality pipeline import failed: {e}. Some features unavailable.")
    DRAFT_JSON_SCHEMA = None  # type: ignore[assignment]
    VERIFIER_JSON_SCHEMA = None  # type: ignore[assignment]

# ── LLM provider (Upgrade 08) ─────────────────────────────────────────────────
# get_llm_provider applies the ALLOW_CLOUD_LLM gate: if the gate is closed or
# the key is absent the factory silently falls back to OllamaProvider and logs
# the reason. Default runtime is always local/Ollama; cloud is strictly opt-in.
from providers.factory import get_llm_provider

_provider = get_llm_provider(cfg)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Use a stable secret key so sessions survive restarts.
# Set SECRET_KEY in .env for production; fall back to a per-boot random key
# only when the env var is absent (dev convenience).
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ENTRY_LENGTH = 1000
TIMEOUT_LEGACY   = 30   # seconds — legacy single-call mode
TIMEOUT_QUALITY  = 90   # seconds — per LLM call in the quality pipeline
BENCHMARK_LATEST_JSON = Path("evals/reports/job_market_patient_model_benchmark_latest.json")

logging.basicConfig(level=logging.INFO)

# ── Vector store (unified retrieval surface) ──────────────────────────────────
# Honors cfg.retrieval_enabled + cfg.vector_backend; returns a NoOpStore when
# retrieval is off so callers can invoke add/query unconditionally.
vector_store = get_vector_store()


# ── Middleware ────────────────────────────────────────────────────────────────
@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


# ── Routes: static pages ──────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return jsonify({
        "status": "ok",
        "version": _get_version(),
        "retrieval": {
            "backend": vector_store.backend_name,
            "enabled": vector_store.enabled,
            "healthy": vector_store.healthcheck(),
        },
    })


@app.route("/models", methods=["GET"])
def get_available_models():
    """Return active + recommended model configuration."""
    selection = get_runtime_model_selection(cfg)
    payload = selection.to_api_payload()
    payload.update({
        "quality_mode_default": cfg.quality_mode_default,
        "retrieval_enabled": cfg.retrieval_enabled,
    })
    return jsonify(payload)


@app.route("/benchmark/latest")
def latest_benchmark():
    """Return the most recent job-market patient model benchmark report."""
    if not BENCHMARK_LATEST_JSON.exists():
        return jsonify({"ok": False, "error": "No model benchmark has been run yet."}), 404

    try:
        with BENCHMARK_LATEST_JSON.open("r", encoding="utf-8") as handle:
            return jsonify({"ok": True, "benchmark": json.load(handle)})
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Could not read benchmark file: %s", exc)
        return jsonify({"ok": False, "error": "Benchmark report could not be read."}), 500


# ── Routes: session ───────────────────────────────────────────────────────────
@app.route("/session/history")
def get_session_history():
    return jsonify(session.get("chat", []))


@app.route("/session/reset", methods=["POST"])
def reset_session():
    session.pop("chat", None)
    return jsonify({"status": "cleared"})


# ── Routes: analysis ──────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze_entry():
    data = request.get_json(silent=True) or {}
    selection = get_runtime_model_selection(cfg)
    journal_entry     = data.get("entry", "").strip()
    model             = data.get("model", selection.generator)
    quality_mode      = data.get("quality_mode", cfg.quality_mode_default)
    baseline_json_mode = data.get("baseline_json_mode", False)

    # ── Input validation ──
    if not journal_entry:
        return jsonify({"error": "✋ Please enter some thoughts before submitting."}), 400
    if len(journal_entry) > MAX_ENTRY_LENGTH:
        return jsonify({"error": f"📏 Entry too long. Limit to {MAX_ENTRY_LENGTH} characters."}), 400
    if not _provider.healthcheck():
        return jsonify({"error": "🛑 The AI assistant is offline. Please try again later."}), 503

    # Retrieval is resolved once here so (a) the prompt context and the
    # `sources` returned to the client are the *same* hits, and (b) the
    # just-submitted entry is excluded by id even though it's written after
    # retrieval (future-proofs any write-before-query reordering).
    namespace = _namespace_for()
    new_id = _new_entry_id()

    try:
        if baseline_json_mode:
            hits = _retrieve_hits(
                journal_entry, namespace=namespace,
                top_k=max(1, cfg.retrieval_top_k - 1), exclude_ids={new_id},
            )
            analysis_json = _run_baseline(
                journal_entry, model, format_hits_as_context(hits)
            )
        elif quality_mode:
            hits = _retrieve_hits(
                journal_entry, namespace=namespace,
                top_k=cfg.retrieval_top_k, exclude_ids={new_id},
            )
            analysis_json = _run_quality_pipeline(
                journal_entry,
                model,
                verifier_model=selection.verifier,
                fallback_model=selection.fallback,
                retrieved_context=format_hits_as_context(hits),
            )
        else:
            return _run_legacy(journal_entry, model, namespace=namespace, entry_id=new_id)

        insight_text = _format_insight(analysis_json)
        _store_in_rag(journal_entry, insight_text, entry_id=new_id, namespace=namespace)
        _append_to_session(journal_entry, insight_text, analysis_json)

        return jsonify({
            "insight": insight_text,
            "analysis": analysis_json,
            "sources": [_hit_to_source(h) for h in hits],
        })

    except ValueError as e:
        err = str(e)
        if "json_parse_failed" in err:
            return jsonify({"error": err}), 500
        logging.exception("ValueError in /analyze")
        return jsonify({"error": "⚠️ Analysis failed. Please try again."}), 500
    except requests.exceptions.Timeout:
        return jsonify({"error": "⏱️ The assistant took too long. Try again."}), 504
    except Exception:
        logging.exception("Unexpected error in /analyze")
        return jsonify({"error": "⚠️ Something unexpected happened. Please try again."}), 500


# ── Routes: prompt suggestion ─────────────────────────────────────────────────
@app.route("/prompt", methods=["POST"])
def suggest_prompt():
    try:
        selection = get_runtime_model_selection(cfg)
        query = (
            "You are a journaling coach. Give the user 1 deep and meaningful self-reflection prompt.\n"
            "Keep it under 20 words. Return just the prompt and nothing else."
        )
        # For the prompt role, use the Anthropic prompt model when on the cloud
        # backend; otherwise the provider's generate() routes to Ollama.
        prompt_model = selection.prompt
        output = _provider.generate(prompt_model, query, timeout=TIMEOUT_LEGACY)
        return jsonify({"prompt": output})
    except Exception:
        logging.exception("Prompt generation failed")
        return jsonify({"error": "Failed to generate a prompt."}), 500


# ── Routes: transcription ─────────────────────────────────────────────────────
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Transcribe voice recording using Whisper.
    Requires `openai-whisper` to be installed (optional dependency).
    Returns 501 with a clear message if Whisper is not available.
    """
    try:
        import whisper  # type: ignore
        import tempfile, io
    except ImportError:
        return jsonify({
            "error": "Transcription requires openai-whisper. Install with: pip install openai-whisper"
        }), 501

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided."}), 400

    audio_bytes = audio_file.read()
    if len(audio_bytes) > cfg.whisper_max_audio_bytes:
        return jsonify({"error": "Audio file too large."}), 413

    try:
        model = whisper.load_model(cfg.whisper_model)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = model.transcribe(tmp_path)
        os.unlink(tmp_path)

        return jsonify({
            "text":     result["text"].strip(),
            "language": result.get("language", "unknown"),
        })
    except Exception:
        logging.exception("Transcription failed")
        return jsonify({"error": "Transcription failed. Please try again."}), 500


# ── Pipeline helpers ──────────────────────────────────────────────────────────
def _run_quality_pipeline(
    journal_entry: str,
    generator_model: str,
    *,
    verifier_model: str,
    fallback_model: str,
    retrieved_context: str = "",
) -> Dict[str, Any]:
    """Draft → Verify → Revise pipeline.

    `retrieved_context` is resolved by the caller (the /analyze route) so the
    same hits ground the prompt and populate the response `sources`.
    """
    # Step 1: Draft — fix #12: validator_model ensures {} never passes through.
    try:
        draft_prompt = get_draft_prompt(journal_entry, retrieved_context)
        draft_json = _provider.json_generate(
            generator_model,
            DRAFT_SYSTEM_PROMPT,
            draft_prompt,
            max_retries=5,
            json_schema=DRAFT_JSON_SCHEMA,
            validator_model=AnalysisOutput,
        )
    except Exception as e:
        logging.error(f"Draft generation failed: {type(e).__name__}")
        raise ValueError("json_parse_failed:stage=draft")

    # Step 2: Verify — fix #12: validator_model enforces VerifierVerdict shape.
    try:
        verifier_prompt = get_verifier_prompt(draft_json, journal_entry, retrieved_context)
        verdict = _provider.json_generate(
            verifier_model,
            VERIFIER_SYSTEM_PROMPT,
            verifier_prompt,
            max_retries=5,
            json_schema=VERIFIER_JSON_SCHEMA,
            validator_model=VerifierVerdict,
        )
        logging.debug(
            f"Verification: groundedness={verdict.get('groundedness_score', 0)}, "
            f"rewrite={verdict.get('rewrite_required', False)}"
        )
    except Exception as e:
        logging.warning(f"Verification failed: {type(e).__name__}. Using draft as-is.")
        verdict = {
            "groundedness_score": 0.5,
            "unsupported_claims": [],
            "safety_flags": [],
            "rewrite_required": False,
            "rewrite_instructions": "",
        }

    # Step 3: Revise if needed
    needs_rewrite = verdict.get("rewrite_required", False)
    groundedness  = verdict.get("groundedness_score", 1.0)

    if needs_rewrite or groundedness < cfg.groundedness_threshold:
        logging.info("Revision required. Calling fallback model.")
        try:
            revision_prompt = get_revision_prompt(
                draft_json, verdict, journal_entry, retrieved_context
            )
            final_json = _provider.json_generate(
                fallback_model,
                DRAFT_SYSTEM_PROMPT,
                revision_prompt,
                max_retries=5,
                json_schema=DRAFT_JSON_SCHEMA,
                validator_model=AnalysisOutput,
            )
            logging.info("Revision completed.")
            return final_json
        except Exception as e:
            logging.error(f"Revision failed: {type(e).__name__}. Using original draft.")
            return draft_json

    logging.info("Draft passed verification.")
    return draft_json


def _run_baseline(
    journal_entry: str,
    generator_model: str,
    retrieved_context: str = "",
) -> Dict[str, Any]:
    """
    Single-pass JSON generation (baseline / eval mode).
    Intentionally weaker than quality mode for DPO pair generation.
    """
    WEAKER_SYSTEM_PROMPT = (
        "You are a journaling assistant. Analyze journal entries and provide structured insights in JSON format.\n\n"
        "Return a JSON object with these fields:\n"
        "- summary: string\n"
        "- emotions: array of strings\n"
        "- patterns: array of strings\n"
        "- triggers: array of strings\n"
        "- coping_suggestions: array of strings (2-3)\n"
        "- quotes_from_user: array of strings (max 3)\n"
        "- confidence: float (0.0-1.0)\n\n"
        "Be helpful and empathetic. Return ONLY valid JSON matching the schema."
    )

    try:
        draft_prompt = get_draft_prompt(journal_entry, retrieved_context)
        # Note defect #14: the baseline path is intentionally weaker (no
        # validator_model, higher temperature, fewer retries) so DPO pairs
        # have a meaningful quality gap between chosen (quality) and rejected
        # (baseline). If the baseline is made as strong as the quality path,
        # preference pairs starve and DPO training degrades.
        return _provider.json_generate(
            generator_model,
            WEAKER_SYSTEM_PROMPT,
            draft_prompt,
            max_retries=3,
            json_schema=DRAFT_JSON_SCHEMA,
            temperature=0.3,
        )
    except Exception as e:
        logging.error(f"Baseline generation failed: {type(e).__name__}")
        raise ValueError("json_parse_failed:stage=baseline_json")


def _run_legacy(
    journal_entry: str,
    model: str,
    *,
    namespace: Optional[str] = None,
    entry_id: Optional[str] = None,
):
    """Simple single-model mode (no structured JSON). Returns a Flask response directly.

    Legacy mode does not inject retrieved context into its prompt, so it reports
    no `sources`; it still indexes the entry for future retrieval.
    """
    safe_entry = html.escape(journal_entry)
    prompt = (
        "You are an emotionally intelligent journaling assistant.\n"
        "When I share a journal entry, analyze it like a therapist.\n"
        "Be concise, warm, and clear.\n"
        "Summarize the core avoided emotion and what it's trying to teach me.\n"
        "Give 2–3 short reflective insights that help me understand myself better.\n"
        "Use gentle, human language. Avoid rambling or restating obvious things.\n\n"
        f"My journal entry:\n\"{safe_entry}\""
    )
    output = _provider.generate(model, prompt, timeout=TIMEOUT_LEGACY)
    for phrase in ["I'm here for you", "Thank you for sharing"]:
        output = output.replace(phrase, "").strip()

    _store_in_rag(journal_entry, output, entry_id=entry_id, namespace=namespace)
    _append_to_session(journal_entry, output, None)
    return jsonify({"insight": output, "sources": []})


# ── Shared helpers: retrieval ─────────────────────────────────────────────────
SNIPPET_LEN = 240  # chars of source text exposed in the API; not the embedding payload.


def _session_id() -> str:
    """Stable per-browser-session id, persisted in the Flask cookie session."""
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_hex(8)
        session["sid"] = sid
        session.modified = True
    return sid


def _namespace_for(session_id: Optional[str] = None) -> str:
    """
    Resolve the retrieval namespace from cfg.rag_namespace_mode.

      session -> session:{sid}      (default; isolates each browser session)
      user    -> user:{header}      (multi-user deployments)
      fixed   -> cfg.rag_namespace_fixed
    """
    mode = cfg.rag_namespace_mode
    if mode == "user":
        uid = request.headers.get(cfg.rag_user_id_header, "anonymous")
        return f"user:{uid}"
    if mode == "fixed":
        return cfg.rag_namespace_fixed
    # default: session
    return f"session:{session_id or _session_id()}"


def _new_entry_id() -> str:
    """Unique id for an entry, precomputed so it can be self-excluded at query time."""
    return f"entry_{datetime.utcnow().isoformat()}_{secrets.token_hex(4)}"


def _retrieve_hits(
    journal_entry: str,
    *,
    namespace: Optional[str] = None,
    top_k: int = 3,
    exclude_ids: Optional[Set[str]] = None,
) -> List["RetrievalHit"]:
    """
    Retrieve grounding hits for `journal_entry` within `namespace`.

    Fix #1: query with the current entry (not "") so hits are relevant.
    Self-exclusion is id-based: a historical entry with identical text but a
    different id is still a legitimate candidate.
    """
    if not vector_store.enabled or not journal_entry:
        return []
    # Over-fetch so dropping the self-id still leaves top_k results.
    raw = vector_store.query(
        journal_entry,
        top_k=top_k + (len(exclude_ids) if exclude_ids else 0),
        namespace=namespace,
    )
    if exclude_ids:
        raw = [h for h in raw if h.id not in exclude_ids]
    return raw[:top_k]


def _retrieve_context(
    journal_entry: str = "",
    top_k: int = 3,
    *,
    namespace: Optional[str] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> str:
    """Backwards-compatible string form (evals, legacy callers)."""
    return format_hits_as_context(
        _retrieve_hits(
            journal_entry, namespace=namespace, top_k=top_k, exclude_ids=exclude_ids
        )
    )


def _hit_to_source(hit: "RetrievalHit") -> Dict[str, Any]:
    """Serialize a RetrievalHit into the audit-friendly /analyze `sources` shape."""
    text = hit.text or ""
    snippet = text[:SNIPPET_LEN] + ("…" if len(text) > SNIPPET_LEN else "")
    return {
        "id": hit.id,
        "score": round(float(hit.score), 4),
        "snippet": snippet,
        "created_at": hit.metadata.get("created_at", ""),
    }


# ── Shared helpers: storage ───────────────────────────────────────────────────
def _store_in_rag(
    entry: str,
    insight: str,
    *,
    entry_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> None:
    """
    Index a journal entry for retrieval.

    The stored text is `entry` only — not the legacy `ENTRY: ... INSIGHT: ...`
    blob. Rationale: we're retrieving *past entries* to ground the current one;
    insights are derivative and shouldn't dominate similarity scoring.
    The insight is kept on the metadata for future audit surfaces.
    """
    if not vector_store.enabled:
        return
    entry_id = entry_id or _new_entry_id()
    vector_store.add_entry(
        entry_id=entry_id,
        text=entry,
        metadata={
            "kind": "journal_entry",
            "namespace": namespace or "",
            "created_at": datetime.utcnow().isoformat(),
            "entry_length": len(entry),
            "insight_preview": (insight or "")[:500],
        },
        namespace=namespace,
    )


def _append_to_session(entry: str, response: str, analysis_json: Optional[Dict]) -> None:
    if "chat" not in session:
        session["chat"] = []
    item: Dict[str, Any] = {"entry": entry, "response": response}
    if analysis_json is not None:
        item["analysis_json"] = analysis_json
    session["chat"].append(item)
    session.modified = True


def _format_insight(analysis_json: Dict[str, Any]) -> str:
    """Format structured JSON into readable plain text (for legacy / non-JS consumers)."""
    parts = []
    if "summary" in analysis_json:
        parts.append(analysis_json["summary"])
    if analysis_json.get("emotions"):
        parts.append(f"\nEmotions: {', '.join(analysis_json['emotions'])}")
    if analysis_json.get("patterns"):
        parts.append(f"\nPatterns: {', '.join(analysis_json['patterns'])}")
    if analysis_json.get("coping_suggestions"):
        suggestions = "\n".join(f"• {s}" for s in analysis_json["coping_suggestions"])
        parts.append(f"\nSuggestions:\n{suggestions}")
    return "\n".join(parts) if parts else json.dumps(analysis_json, indent=2)


def _get_version() -> str:
    try:
        from version import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    app.run(debug=True)
