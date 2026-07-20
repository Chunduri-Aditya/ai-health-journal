# === app.py ===
# AI Health Journal — Flask backend
# Multi-model quality pipeline with RAG, session tracking, transcription support.

from __future__ import annotations

import html
import json
import logging
import os
import re
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
from privacy.redact import redact

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

# ── Crisis gate ───────────────────────────────────────────────────────────────
# A deterministic floor beneath the verifier's judgment. Reflexive, first-person
# self-harm phrasing forces the support path even if the verifier misses it or
# its call failed, so the reframe/positivity path fails closed. Kept tight (first
# person, reflexive) so idioms like "this job is killing me" do not trigger it.
#
# Coverage is asymmetric on purpose: a false positive here just shows a
# supportive message on a non-crisis entry (mildly imprecise); a false
# negative lets positivity/reframe reach a genuine crisis. So this errs
# toward catching more real-world phrasing, while still requiring first-
# person/reflexive structure to avoid firing on idioms or third-person
# statements. Known gap, intentionally NOT covered: vaguer phrasing like
# "I won't be here much longer" is too easily a benign statement (moving,
# changing jobs) to regex safely -- that's left to the verifier's judgment.
# Non-English phrasing is also not covered here; see verifier_prompts.py for
# the verifier-side mitigation.
#
# Two additions were tried and dropped after live false-positive testing:
# bare "kms" collides with the common tech acronym (AWS KMS -- a real risk
# for exactly this app's job/work-stress journaling use case), and bare
# "overdose" collides with the ordinary idiom "an overdose of X". Kept
# instead: the more specific pills-object pattern below. One known accepted
# false-positive tradeoff we DID keep: "jumping off a bridge" also matches
# benign bungee-jumping journal entries -- kept anyway because catching
# method-specific ideation outweighs an occasional unnecessary supportive
# message (see the asymmetric-risk note above).
_CRISIS_PATTERNS = re.compile(
    r"\b("
    r"kill(?:ing)?\s+myself|"
    r"end(?:ing)?\s+my\s+life|"
    r"end(?:ing)?\s+it\s+all|"
    r"take\s+my\s+own\s+life|"
    r"want(?:s|ed)?\s+to\s+die|"
    r"wish\s+(?:i\s+was|i\s+were)\s+dead|"
    r"better\s+off\s+(?:if\s+i\s+(?:was|were)\s+)?dead|"
    r"give\s+up\s+on\s+life|"
    r"(?:no\s+longer|don'?t)\s+want(?:s)?\s+to\s+exist|"
    r"don'?t\s+want\s+to\s+(?:live|be\s+here|be\s+alive|wake\s+up)|"
    r"hurt(?:ing)?\s+myself|harm(?:ing)?\s+myself|"
    r"unalive(?:\s+myself)?|"
    r"self[-\s]?harm|"
    r"suicid(?:e|al)|"
    r"overdos(?:e|ed|ing)\s+on\s+(?:my|the)\s+(?:pills|meds|medication)|"
    r"tak(?:e|ing)\s+all\s+(?:my|the)\s+pills|"
    r"jump(?:ing)?\s+off\s+(?:a|the)\s+bridge"
    r")\b",
    re.IGNORECASE,
)

# Deterministic floor for explicit, first-person, PLANNED intent to harm
# someone else. Kept much narrower than _CRISIS_PATTERNS above: "I'm going to
# kill him" / "he's going to kill me" is extremely common non-literal
# hyperbole in everyday venting (annoyance, sibling teasing, sports trash
# talk) in a way "I want to kill myself" essentially never is, so a bare verb
# match here would false-positive constantly. Requires an explicit first-
# person intent marker ("I'm going to" / "I'm planning to" / "I'll" or
# "I will") directly attached to the harm clause -- catches deliberate
# threats, not "that really hurt him badly" (third-person, emotional) or
# "I'm going to kill him" (hyperbole, no specific harm verb + intensifier).
_HARM_TO_OTHERS_PATTERNS = re.compile(
    r"\b("
    r"i(?:'m| am)\s+(?:going\s+to|planning\s+to)\s+(?:hurt|attack|harm)\s+(?:him|her|them)\b|"
    r"i(?:'ll| will)\s+hurt\s+(?:him|her|them)\s+(?:badly|seriously|for\s+real)"
    r")\b",
    re.IGNORECASE,
)

# Elevated distress that is NOT crisis: hopelessness, worthlessness, harsh
# self-blame. This is the tier between ordinary sadness (handled by a normal
# reframe) and self-harm (handled by _CRISIS_PATTERNS above). Firing here only
# adds a steadying acknowledgement -- it never suppresses the analysis -- so the
# cost of a false positive is one extra kind sentence, which is why this can
# afford to be broader than the crisis floor.
#
# Deliberately first-person: "you're worthless" (quoting someone else) or "the
# project is a failure" are not the user judging themselves. "i'm a burden" is
# included even though it sits close to crisis phrasing -- it is a recognized
# risk marker, and routing it to a supportive message is the safe direction.
_DISTRESS_PATTERNS = re.compile(
    r"\b("
    r"i(?:'m| am)\s+(?:such\s+)?(?:a\s+)?(?:failure|worthless|useless|broken|pathetic|a\s+burden)|"
    # "I feel like a failure" is at least as common as "I am a failure" and was
    # missed by the copula-only form above.
    r"i\s+feel\s+like\s+(?:such\s+)?(?:a\s+)?(?:failure|burden|nothing)|"
    r"i\s+feel\s+(?:worthless|useless|broken|pathetic)|"
    r"i\s+hate\s+myself|"
    r"i(?:'m| am)\s+not\s+good\s+enough|"
    r"i\s+can'?t\s+do\s+anything\s+right|"
    r"i(?:'ll| will)\s+never\s+get\s+(?:better|past\s+this)|"
    r"nothing\s+(?:i\s+do\s+)?matters|"
    r"what'?s\s+the\s+point|"
    r"no\s+one\s+(?:cares|would\s+notice)|"
    r"i\s+(?:always|constantly)\s+(?:ruin|screw\s+up|mess\s+up)"
    r")\b",
    re.IGNORECASE,
)

# Reported speech guard. "My friend said I am worthless" is the user describing
# someone else's cruelty, not judging themselves, and answering it with "that
# sounds hard about yourself" would misread the entry. Only third-party subjects
# are listed: "I said I'm a failure" is still the user's own self-judgment and
# must keep firing.
_REPORTED_SPEECH = re.compile(
    r"\b(?:he|she|they|someone|everyone|nobody|people|my\s+\w+|his|her|their)\s+"
    r"(?:said|says|told\s+me|calls?\s+me|called\s+me|thinks?|thought)\b[^.!?]{0,25}$",
    re.IGNORECASE,
)

# Language the ASSISTANT must never use back at the user. Blame, dismissal, and
# clinical labelling are the three failure modes that turn a journaling reply
# into something that lands badly on someone already struggling.
#
# Kept narrow on purpose: ordinary directive phrasing ("consider talking to your
# manager", "you might try") is the product working correctly, so this matches
# only blame/dismissal/diagnosis, never advice as such. The asymmetry runs the
# other way from the crisis gate: a false positive here silently drops one
# useful suggestion, so over-broad patterns would quietly gut the analysis.
_HARSH_OUTPUT_PATTERNS = re.compile(
    r"("
    # Blame and command-shame
    r"\byou\s+(?:should|need\s+to|have\s+to|must)\s+(?:just\s+)?(?:stop|quit|get\s+over)\b|"
    r"\bjust\s+(?:get\s+over|move\s+on|snap\s+out)\b|"
    r"\byou(?:'re| are)\s+(?:being\s+)?(?:irrational|dramatic|lazy|weak|childish|ridiculous|overreacting)\b|"
    r"\byour\s+own\s+fault\b|\byou\s+brought\s+this\s+on\s+yourself\b|"
    r"\bstop\s+(?:being|feeling|complaining|whining)\b|"
    # Dismissal / minimisation
    r"\bit'?s\s+not\s+(?:that\s+)?bad\b|"
    r"\bothers\s+have\s+it\s+worse\b|\bcould\s+be\s+worse\b|"
    r"\bstop\s+overthinking\b|"
    # Clinical labelling of the person (diagnosis is out of scope for this app)
    r"\byou\s+(?:have|suffer\s+from)\s+(?:depression|anxiety|a\s+disorder|bipolar|ptsd)\b|"
    r"\byou(?:'re| are)\s+(?:clinically\s+)?(?:depressed|mentally\s+ill)\b"
    r")",
    re.IGNORECASE,
)

# Product-owner editable. Shown INSTEAD of a reframe when the crisis gate fires.
# Acknowledges without diagnosing and points outward. Set AIHJ_CRISIS_MESSAGE to
# localize the resource line for your region.
# `or` (not getenv's default= param) so an explicitly BLANK env var
# (AIHJ_CRISIS_MESSAGE="", a plausible operator misconfiguration -- vs.
# simply leaving it unset) still falls back to the built-in message instead
# of silently rendering a crisis entry with no support text and no reframe
# at all. getenv's default= only applies when the var is absent, not when
# it's present-but-empty.
CRISIS_SUPPORT_MESSAGE = os.getenv("AIHJ_CRISIS_MESSAGE") or (
    "It sounds like you're carrying something really heavy right now, and you "
    "don't have to carry it alone. Please consider reaching out to a crisis line "
    "in your area or someone you trust. If you're in immediate danger, contact "
    "local emergency services."
)

# Shown ABOVE the analysis when the distress gate fires. Deliberately does NOT
# point outward to services the way CRISIS_SUPPORT_MESSAGE does: this tier is
# ordinary human difficulty, not an emergency, and medicalising it would be both
# inaccurate and alienating. It acknowledges and steadies, nothing more.
DISTRESS_STEADYING_MESSAGE = os.getenv("AIHJ_STEADYING_MESSAGE") or (
    "That sounds genuinely hard, and it makes sense that it's sitting heavily "
    "with you. Nothing below is a verdict on you. It's a reflection of what you "
    "wrote, so take it at whatever pace feels right."
)

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
    data = request.get_json(silent=True)
    # get_json(silent=True) returns None on unparseable JSON, and a bare
    # value (list, str, number) on valid-but-non-object bodies. Both must be
    # rejected here, before any .get()/.strip() call, or a malformed/wrong-
    # typed client request crashes with an unhandled AttributeError instead
    # of a clean 400 (Flask's bare default error page leaks internal detail
    # in debug mode, and is the wrong content-type for a JSON API either way).
    if not isinstance(data, dict):
        return jsonify({"error": "There's nothing to look at yet. Write a little first."}), 400

    entry = data.get("entry", "")
    if not isinstance(entry, str):
        return jsonify({"error": "There's nothing to look at yet. Write a little first."}), 400
    journal_entry     = entry.strip()
    selection = get_runtime_model_selection(cfg)
    model             = data.get("model", selection.generator)
    quality_mode      = data.get("quality_mode", cfg.quality_mode_default)
    baseline_json_mode = data.get("baseline_json_mode", False)

    # ── Input validation ──
    if not journal_entry:
        return jsonify({"error": "There's nothing to look at yet. Write a little first."}), 400
    if len(journal_entry) > MAX_ENTRY_LENGTH:
        return jsonify({"error": f"That's a bit longer than this can take in at once. Try trimming to {MAX_ENTRY_LENGTH} characters."}), 400
    if not _provider.healthcheck():
        return jsonify({"error": "Your journal is here, but the reflection service isn't responding right now. Your writing is safe."}), 503

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
            # The internal stage tag (e.g. "json_parse_failed:stage=draft")
            # is logged for debugging but never returned to the client --
            # it's implementation detail, not something a user needs, and
            # every other branch here already uses a generic message.
            logging.error(f"Pipeline JSON parse failure: {err}")
            return jsonify({"error": "That didn't come through. Your entry is still here, so feel free to try again."}), 500
        logging.exception("ValueError in /analyze")
        return jsonify({"error": "That didn't come through. Your entry is still here, so feel free to try again."}), 500
    except requests.exceptions.Timeout:
        return jsonify({"error": "That took longer than expected and timed out. Your entry is still here."}), 504
    except Exception:
        logging.exception("Unexpected error in /analyze")
        return jsonify({"error": "Something went wrong on our side, not yours. Your entry is still here."}), 500


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
        return jsonify({"error": "Couldn't fetch a prompt just now. You can always start with whatever is on your mind."}), 500


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
        return jsonify({"error": "Couldn't make out the recording. You can try again or type instead."}), 500


# ── Crisis gate ───────────────────────────────────────────────────────────────
def _is_crisis(journal_entry: str, verdict: Dict[str, Any]) -> bool:
    """Crisis decision for the reframe gate.

    Fails closed: fires if the verifier judged a crisis, if any safety flag
    names self-harm or harm to others, OR if the raw entry matches the
    reflexive self-harm floor or the (narrower) harm-to-others floor. The
    floors cover the case where the verifier call failed or missed it.
    """
    if verdict.get("crisis_detected"):
        return True
    # Normalize separators so underscore/hyphen-styled flags ("self_harm_
    # ideation") match the same as space-separated ones ("self harm
    # ideation") -- the verifier's phrasing of a flag is not something we
    # control, and a bare exact-substring match on 4 hardcoded strings missed
    # every paraphrase tested (see tests/test_crisis_gate_adversarial.py).
    flags = " ".join(verdict.get("safety_flags", [])).lower().replace("_", " ").replace("-", " ")
    crisis_flag_terms = (
        "self harm", "suicid", "harm to self", "danger to self", "self injury",
        "harm ideation", "harm to others", "danger to others", "threat to others",
    )
    if any(term in flags for term in crisis_flag_terms):
        return True
    if _CRISIS_PATTERNS.search(journal_entry or ""):
        return True
    return bool(_HARM_TO_OTHERS_PATTERNS.search(journal_entry or ""))


def _is_distressed(journal_entry: str) -> bool:
    """Elevated distress that is not crisis: hopelessness, worthlessness, self-blame.

    Entry-text only, unlike _is_crisis: this tier exists to acknowledge how the
    user is speaking about themselves, which is visible in the raw text and does
    not need the verifier's judgment. Matches attributed to someone else are
    skipped (see _REPORTED_SPEECH).
    """
    text = journal_entry or ""
    for match in _DISTRESS_PATTERNS.finditer(text):
        if _REPORTED_SPEECH.search(text[: match.start()]):
            continue
        return True
    return False


def _apply_reframe_gate(
    analysis_json: Dict[str, Any],
    journal_entry: str,
    verdict: Dict[str, Any],
) -> Dict[str, Any]:
    """Route the entry to the right emotional register: crisis, distress, or normal.

    Deterministic by design: the model classifies (crisis_detected), code acts.
    The three tiers are mutually exclusive and ordered by severity:

      crisis   -> clear the reframe, attach a support message pointing outward.
      distress -> keep the reframe (a gentle one is appropriate here) and add a
                  steadying acknowledgement above the analysis.
      normal   -> untouched.

    Crisis wins over distress: a self-harm entry must never be answered with the
    softer "that sounds hard" framing when it needs the support pathway.
    """
    analysis_json.setdefault("crisis_support", False)
    analysis_json.setdefault("support_message", "")
    analysis_json.setdefault("distress_support", False)
    analysis_json.setdefault("steadying_message", "")

    if _is_crisis(journal_entry, verdict):
        analysis_json["reframe"] = ""
        analysis_json["crisis_support"] = True
        analysis_json["support_message"] = CRISIS_SUPPORT_MESSAGE
    elif _is_distressed(journal_entry):
        analysis_json["distress_support"] = True
        analysis_json["steadying_message"] = DISTRESS_STEADYING_MESSAGE
    return analysis_json


# Fields the user actually reads. `quotes_from_user` is excluded on purpose: it
# echoes the user's own words back, so harsh phrasing there is the user's, not
# the assistant's, and stripping it would censor the person's own journal.
_USER_FACING_TEXT_FIELDS = ("summary", "reframe", "support_message", "steadying_message")
_USER_FACING_LIST_FIELDS = ("patterns", "coping_suggestions", "journaling_feedback", "triggers")


def _find_harsh_content(analysis_json: Dict[str, Any]) -> List[str]:
    """Return every assistant-authored snippet that reads as blaming, dismissive,
    or clinically labelling. Empty list means the tone is acceptable."""
    offenders: List[str] = []
    for field in _USER_FACING_TEXT_FIELDS:
        value = analysis_json.get(field)
        if isinstance(value, str) and _HARSH_OUTPUT_PATTERNS.search(value):
            offenders.append(f"{field}: {value}")
    for field in _USER_FACING_LIST_FIELDS:
        for item in analysis_json.get(field) or []:
            if isinstance(item, str) and _HARSH_OUTPUT_PATTERNS.search(item):
                offenders.append(f"{field}: {item}")
    return offenders


def _strip_harsh_items(analysis_json: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic last line of defence on assistant tone.

    Drops harsh items from the list fields outright, mirroring
    _strip_ungrounded_quotes: whether a sentence blames or dismisses the user is
    checkable in code, so it does not depend on a small local model reliably
    policing its own output. Runs AFTER the revise step, so the collaborative
    pipeline gets first attempt at rewriting the tone properly and this only
    catches what survived.

    `summary` is deliberately not stripped -- it is required and non-empty, so
    removing it would break the schema. A harsh summary is instead forced
    through the revise step by _run_quality_pipeline before reaching here.
    """
    for field in _USER_FACING_LIST_FIELDS:
        items = analysis_json.get(field) or []
        kept = [i for i in items if not (isinstance(i, str) and _HARSH_OUTPUT_PATTERNS.search(i))]
        if len(kept) != len(items):
            logging.warning(
                f"Dropped {len(items) - len(kept)} harsh item(s) from '{field}' before display"
            )
            analysis_json[field] = kept
    return analysis_json


def _strip_ungrounded_quotes(analysis_json: Dict[str, Any], journal_entry: str) -> Dict[str, Any]:
    """Drop any quotes_from_user entry that isn't an exact substring of the
    journal entry.

    "Is this quote actually in the entry" needs no LLM judgment at all -- it
    is 100% mechanically checkable, and the verifier cannot be trusted to
    catch it alone (live-confirmed: a fabricated quote with zero grounding in
    an off-topic entry passed verification with groundedness_score=0.95, see
    tests/test_prompt_injection_adversarial.py). Deterministic logic in code
    for a checkable property, not judgment in the model.
    """
    quotes = analysis_json.get("quotes_from_user") or []
    grounded = [q for q in quotes if q in journal_entry]
    dropped = [q for q in quotes if q not in journal_entry]
    if dropped:
        logging.warning(f"Dropped {len(dropped)} ungrounded quote(s) not present in the entry")
    analysis_json["quotes_from_user"] = grounded
    return analysis_json


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

    draft_json = _strip_ungrounded_quotes(draft_json, journal_entry)

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

    # Step 2b: Tone floor. A blaming or dismissive draft is forced back through
    # the revise step even when the verifier judged it fine -- the verifier is a
    # small local model and tone is exactly the thing it misses quietly. Merged
    # into the verdict (rather than handled separately) so the revision prompt
    # carries the real reason and the fallback model can fix it properly.
    harsh = _find_harsh_content(draft_json)
    if harsh:
        logging.warning(f"Harsh tone detected in draft: {harsh}")
        verdict["safety_flags"] = list(verdict.get("safety_flags", [])) + [
            "harsh or dismissive tone toward the user"
        ]
        verdict["rewrite_required"] = True
        verdict["rewrite_instructions"] = (
            (verdict.get("rewrite_instructions") or "").strip()
            + " Rewrite so nothing blames, dismisses, or diagnoses the user. "
            "Acknowledge the difficulty plainly and keep suggestions gentle and optional."
        ).strip()

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
            final_json = _strip_ungrounded_quotes(final_json, journal_entry)
            final_json = _strip_harsh_items(final_json)
            return _apply_reframe_gate(final_json, journal_entry, verdict)
        except Exception as e:
            logging.error(f"Revision failed: {type(e).__name__}. Using original draft.")
            return _apply_reframe_gate(_strip_harsh_items(draft_json), journal_entry, verdict)

    logging.info("Draft passed verification.")
    draft_json = _strip_harsh_items(draft_json)
    return _apply_reframe_gate(draft_json, journal_entry, verdict)


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
def _maybe_redact(text: str) -> str:
    """Scrub PII before it is persisted anywhere, when PRIVACY_MODE=strict.

    Used by both the RAG store (_store_in_rag) and the client-side session
    (_append_to_session) -- the session cookie is signed but NOT encrypted
    (readable via base64 by anyone with cookie access: devtools, an
    extension, XSS, a non-HTTPS hop), so it needs the same treatment as the
    RAG store, not just a separate "it's local disk" trust boundary.

    Only `strict` redacts; any other mode (default `balanced`) stores raw
    text, preserving current behavior. Redaction narrows exposure to the
    categories redact() actually catches (email, US-format phone) -- it does
    not encrypt the cookie itself, so PII categories redact() doesn't cover
    are still exposed in strict mode too.
    """
    if cfg.privacy_mode == "strict":
        return redact(text)
    return text


def _store_in_rag(
    entry: str,
    insight: str,
    *,
    entry_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> bool:
    """
    Index a journal entry for retrieval.

    The stored text is `entry` only — not the legacy `ENTRY: ... INSIGHT: ...`
    blob. Rationale: we're retrieving *past entries* to ground the current one;
    insights are derivative and shouldn't dominate similarity scoring.
    The insight is kept on the metadata for future audit surfaces.

    Returns True if the entry was actually indexed, False if the write
    failed or retrieval is disabled. A write failure never raises here (the
    LLM analysis already succeeded and must still reach the user; a failed
    RAG write is a secondary concern) but it also must not be indistinguish-
    able from success — the caller can check this to decide whether to warn
    the user, retry, or just rely on the log line this also emits.
    """
    if not vector_store.enabled:
        return False
    entry_id = entry_id or _new_entry_id()
    # PRIVACY_MODE=strict scrubs PII (emails, phones) before it is persisted to
    # the local RAG store, so the retrievable history and its metadata are
    # redacted at rest. Mode change is not retroactive: entries written under a
    # looser mode keep their original text. See _maybe_redact.
    stored_text = _maybe_redact(entry)
    ok = vector_store.add_entry(
        entry_id=entry_id,
        text=stored_text,
        metadata={
            "kind": "journal_entry",
            "namespace": namespace or "",
            "created_at": datetime.utcnow().isoformat(),
            "entry_length": len(stored_text),
            "insight_preview": _maybe_redact(insight or "")[:500],
        },
        namespace=namespace,
    )
    if not ok:
        logging.warning(f"RAG write failed for entry_id={entry_id}; not indexed for retrieval")
    return ok


def _append_to_session(entry: str, response: str, analysis_json: Optional[Dict]) -> None:
    """Persist one turn into the client-side session (history sidebar).

    Routed through _maybe_redact for the same reason _store_in_rag is: the
    session cookie is signed but not encrypted, so under PRIVACY_MODE=strict
    it must not carry raw PII either. analysis_json is stored as-is (not
    deep-redacted) -- matches the scope already established for the RAG
    store's metadata, which only redacts the flat entry/insight text too.
    """
    if "chat" not in session:
        session["chat"] = []
    item: Dict[str, Any] = {
        "entry": _maybe_redact(entry),
        "response": _maybe_redact(response),
    }
    if analysis_json is not None:
        item["analysis_json"] = analysis_json
    session["chat"].append(item)
    session.modified = True


def _format_insight(analysis_json: Dict[str, Any]) -> str:
    """Format structured JSON into readable plain text (for legacy / non-JS consumers)."""
    parts = []
    # Steadying acknowledgement leads, so the first thing read on a hard entry is
    # not a clinical breakdown of it.
    if analysis_json.get("distress_support") and analysis_json.get("steadying_message"):
        parts.append(analysis_json["steadying_message"])
    if "summary" in analysis_json:
        parts.append(analysis_json["summary"])
    if analysis_json.get("emotions"):
        parts.append(f"\nEmotions: {', '.join(analysis_json['emotions'])}")
    if analysis_json.get("patterns"):
        parts.append(f"\nPatterns: {', '.join(analysis_json['patterns'])}")
    if analysis_json.get("coping_suggestions"):
        suggestions = "\n".join(f"• {s}" for s in analysis_json["coping_suggestions"])
        parts.append(f"\nSuggestions:\n{suggestions}")
    if analysis_json.get("journaling_feedback"):
        tips = "\n".join(f"• {s}" for s in analysis_json["journaling_feedback"])
        parts.append(f"\nJournaling tips:\n{tips}")
    # Crisis support and reframe are mutually exclusive: the gate clears reframe
    # on crisis, so support_message never appears alongside positivity.
    if analysis_json.get("crisis_support") and analysis_json.get("support_message"):
        parts.append(f"\n{analysis_json['support_message']}")
    elif analysis_json.get("reframe"):
        parts.append(f"\nA gentler way to see this:\n{analysis_json['reframe']}")
    return "\n".join(parts) if parts else json.dumps(analysis_json, indent=2)


def _get_version() -> str:
    try:
        from version import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    # This is the app's actual launch path (make run / start.sh both run
    # `python app.py`), not a dev-only branch -- debug must default OFF so an
    # unhandled exception never returns the full Werkzeug debugger (source,
    # local variables, a code-execution console) to whoever can reach the
    # port. Opt in explicitly for local development: FLASK_DEBUG=true.
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
