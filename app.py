# === Enhanced app.py ===
# Supports journaling prompt generation, chat history display, session tracking
# Now includes multi-model quality pipeline with RAG

from flask import Flask, request, jsonify, render_template, session
import requests
import logging
import html
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import new modules
try:
    from llm_client import ollama_generate, json_generate, check_ollama_available, DRAFT_JSON_SCHEMA, VERIFIER_JSON_SCHEMA
    from generator_prompts import DRAFT_SYSTEM_PROMPT, get_draft_prompt
    from verifier_prompts import VERIFIER_SYSTEM_PROMPT, get_verifier_prompt
    from rag_store import get_rag_store
except ImportError as e:
    logging.warning(f"Failed to import quality pipeline modules: {e}. Some features may be unavailable.")
    # Fallback for basic functionality
    def check_ollama_available():
        try:
            import requests
            response = requests.get("http://localhost:11434", timeout=2)
            return response.status_code == 200
        except:
            return False
    DRAFT_JSON_SCHEMA = None
    VERIFIER_JSON_SCHEMA = None

app = Flask(__name__)
app.secret_key = os.urandom(24)

# === CONFIG ===
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Model configuration (from env or defaults)
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "phi3:3.8b")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "phi3:3.8b")  # Can use same or different
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "samantha-mistral:7b")
PROMPT_MODEL = os.getenv("PROMPT_MODEL", "samantha-mistral:7b")
DEFAULT_MODEL = GENERATOR_MODEL  # Backward compatibility

# Feature flags
QUALITY_MODE_DEFAULT = os.getenv("QUALITY_MODE_DEFAULT", "false").lower() == "true"
RETRIEVAL_ENABLED = os.getenv("RETRIEVAL_ENABLED", "true").lower() == "true"
GROUNDEDNESS_THRESHOLD = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.75"))

MAX_LENGTH = 1000
TIMEOUT_SECONDS = 30

logging.basicConfig(level=logging.INFO)

# Initialize RAG store
rag_store = get_rag_store(enabled=RETRIEVAL_ENABLED)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

def analyze_with_quality_pipeline(journal_entry: str, generator_model: str) -> dict:
    """
    Draft → Verify → Revise pipeline for high-quality analysis.
    
    Returns:
        Final analysis JSON with all required fields
    """
    # Step 1: Retrieve context (RAG)
    retrieved_context = ""
    if RETRIEVAL_ENABLED and rag_store.enabled:
        retrieved_context = rag_store.retrieve(journal_entry, top_k=3)
        logging.debug(f"Retrieved {len(retrieved_context)} chars of context")
    
    # Step 2: Draft generation
    try:
        draft_prompt = get_draft_prompt(journal_entry, retrieved_context)
        draft_json = json_generate(
            generator_model,
            DRAFT_SYSTEM_PROMPT,
            draft_prompt,
            max_retries=5,
            json_schema=DRAFT_JSON_SCHEMA
        )
        logging.debug("Draft generated successfully")
    except Exception as e:
        logging.error(f"Draft generation failed: {type(e).__name__}")
        raise ValueError(f"json_parse_failed:stage=draft")
    
    # Step 3: Verification
    try:
        verifier_prompt = get_verifier_prompt(draft_json, journal_entry, retrieved_context)
        verdict = json_generate(
            VERIFIER_MODEL,
            VERIFIER_SYSTEM_PROMPT,
            verifier_prompt,
            max_retries=5,
            json_schema=VERIFIER_JSON_SCHEMA
        )
        logging.debug(f"Verification: groundedness={verdict.get('groundedness_score', 0)}, rewrite_required={verdict.get('rewrite_required', False)}")
    except Exception as e:
        logging.warning(f"Verification failed: {type(e).__name__}. Proceeding with draft.")
        verdict = {
            "groundedness_score": 0.5,
            "unsupported_claims": [],
            "safety_flags": [],
            "rewrite_required": False,
            "rewrite_instructions": ""
        }
    
    # Step 4: Revise if needed
    rewrite_required = verdict.get("rewrite_required", False)
    groundedness_score = verdict.get("groundedness_score", 1.0)
    
    if rewrite_required or groundedness_score < GROUNDEDNESS_THRESHOLD:
        logging.info("Revision required. Calling fallback model.")
        try:
            revision_prompt = f"""Revise this analysis based on verification feedback.

ORIGINAL DRAFT:
{json.dumps(draft_json, indent=2)}

VERIFICATION FEEDBACK:
- Groundedness Score: {groundedness_score}
- Unsupported Claims: {verdict.get('unsupported_claims', [])}
- Safety Flags: {verdict.get('safety_flags', [])}
- Instructions: {verdict.get('rewrite_instructions', 'Fix unsupported claims and ensure all information is grounded in evidence.')}

CURRENT ENTRY:
{journal_entry}

RETRIEVED CONTEXT:
{retrieved_context if retrieved_context else '(none)'}

Return the revised JSON with the same structure as the original draft. Ensure all claims are supported by evidence."""
            
            final_json = json_generate(
                FALLBACK_MODEL,
                DRAFT_SYSTEM_PROMPT,
                revision_prompt,
                max_retries=5,
                json_schema=DRAFT_JSON_SCHEMA
            )
            logging.info("Revision completed")
            return final_json
        except Exception as e:
            logging.error(f"Revision failed: {type(e).__name__}. Using original draft.")
            return draft_json
    else:
        logging.info("Draft passed verification. No revision needed.")
        return draft_json

def format_insight_from_json(analysis_json: dict) -> str:
    """
    Format JSON analysis into readable insight text for UI.
    Maintains backward compatibility with simple text output.
    """
    parts = []
    
    if "summary" in analysis_json:
        parts.append(analysis_json["summary"])
    
    if "emotions" in analysis_json and analysis_json["emotions"]:
        parts.append(f"\nEmotions: {', '.join(analysis_json['emotions'])}")
    
    if "patterns" in analysis_json and analysis_json["patterns"]:
        parts.append(f"\nPatterns: {', '.join(analysis_json['patterns'])}")
    
    if "coping_suggestions" in analysis_json and analysis_json["coping_suggestions"]:
        parts.append(f"\nSuggestions:\n" + "\n".join(f"• {s}" for s in analysis_json["coping_suggestions"]))
    
    return "\n".join(parts) if parts else json.dumps(analysis_json, indent=2)

def analyze_with_baseline_json(journal_entry: str, generator_model: str) -> dict:
    """
    Single-pass JSON generation (baseline_json mode).
    INTENTIONALLY WEAKER than quality mode:
    - No verify/revise
    - Higher temperature (more drift)
    - Simpler system prompt (fewer guardrails)
    - Still uses schema for comparability
    
    Returns:
        Analysis JSON with all required fields
    """
    # Retrieve context (RAG) if enabled (but use fewer for baseline)
    retrieved_context = ""
    if RETRIEVAL_ENABLED and rag_store.enabled:
        retrieved_context = rag_store.retrieve(journal_entry, top_k=2)  # Fewer than quality (3)
        logging.debug(f"Retrieved {len(retrieved_context)} chars of context")
    
    # Weaker system prompt (fewer guardrails, less emphasis on safety)
    WEAKER_SYSTEM_PROMPT = """You are a journaling assistant. Analyze journal entries and provide structured insights in JSON format.

Return a JSON object with these fields:
- summary: string (emotional summary)
- emotions: array of strings (emotions detected)
- patterns: array of strings (behavioral patterns observed)
- triggers: array of strings (potential triggers mentioned)
- coping_suggestions: array of strings (2-3 suggestions)
- quotes_from_user: array of strings (phrases from entry, max 3)
- confidence: float (0.0-1.0, confidence in analysis)

Be helpful and empathetic. Return ONLY valid JSON matching the schema."""
    
    # Single-pass generation with schema enforcement but weaker prompt
    try:
        draft_prompt = get_draft_prompt(journal_entry, retrieved_context)
        # Use weaker system prompt and allow higher temperature (more drift/invention risk)
        analysis_json = json_generate(
            generator_model,
            WEAKER_SYSTEM_PROMPT,  # Weaker prompt
            draft_prompt,
            max_retries=3,  # Fewer retries
            json_schema=DRAFT_JSON_SCHEMA,  # Still enforce schema for comparability
            temperature=0.3  # Higher temperature for baseline (more variation, more invention risk)
        )
        logging.debug("Baseline JSON generated successfully")
        return analysis_json
    except Exception as e:
        logging.error(f"Baseline JSON generation failed: {type(e).__name__}")
        raise ValueError(f"json_parse_failed:stage=baseline_json")


@app.route("/analyze", methods=["POST"])
def analyze_entry():
    data = request.get_json()
    journal_entry = data.get("entry", "").strip()
    model = data.get("model", DEFAULT_MODEL)
    quality_mode = data.get("quality_mode", QUALITY_MODE_DEFAULT)
    baseline_json_mode = data.get("baseline_json_mode", False)

    if not journal_entry:
        return jsonify({"error": "✋ Please enter some thoughts before submitting."}), 400
    if len(journal_entry) > MAX_LENGTH:
        return jsonify({"error": f"📏 Entry too long. Limit to {MAX_LENGTH} characters."}), 400

    if not check_ollama_available():
        return jsonify({"error": "🛑 The AI assistant is offline. Please try again later."}), 503

    try:
        if baseline_json_mode:
            # Baseline JSON mode: single-pass JSON generation (no verify/revise)
            try:
                analysis_json = analyze_with_baseline_json(journal_entry, model)
            except ValueError as e:
                error_msg = str(e)
                if "json_parse_failed" in error_msg:
                    return jsonify({"error": error_msg}), 500
                raise
            
            insight_text = format_insight_from_json(analysis_json)
            
            if RETRIEVAL_ENABLED and rag_store.enabled:
                rag_store.add_entry(journal_entry, insight_text)
            
            if "chat" not in session:
                session["chat"] = []
            session["chat"].append({
                "entry": journal_entry,
                "response": insight_text,
                "analysis_json": analysis_json
            })
            
            return jsonify({
                "insight": insight_text,
                "analysis": analysis_json
            })
        elif quality_mode:
            # New quality pipeline: Draft → Verify → Revise
            try:
                analysis_json = analyze_with_quality_pipeline(journal_entry, model)
            except ValueError as e:
                # JSON parse failure - return error, don't fall back to legacy
                error_msg = str(e)
                if "json_parse_failed" in error_msg:
                    return jsonify({"error": error_msg}), 500
                raise
            
            # Format for UI (backward compatible)
            insight_text = format_insight_from_json(analysis_json)
            
            # Store in RAG if enabled
            if RETRIEVAL_ENABLED and rag_store.enabled:
                rag_store.add_entry(journal_entry, insight_text)
            
            # Save to session history
            if "chat" not in session:
                session["chat"] = []
            session["chat"].append({
                "entry": journal_entry,
                "response": insight_text,
                "analysis_json": analysis_json  # Store full JSON for future use
            })
            
            # Return both formatted text and JSON
            return jsonify({
                "insight": insight_text,
                "analysis": analysis_json  # Full structured analysis
            })
        else:
            # Legacy simple mode (backward compatible)
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
            
            output = ollama_generate(model, prompt, timeout=TIMEOUT_SECONDS)
            for phrase in ["I'm here for you", "Thank you for sharing"]:
                output = output.replace(phrase, "").strip()
            
            # Save to session history
            if "chat" not in session:
                session["chat"] = []
            session["chat"].append({"entry": journal_entry, "response": output})
            
            # Store in RAG if enabled
            if RETRIEVAL_ENABLED and rag_store.enabled:
                rag_store.add_entry(journal_entry, output)
            
            return jsonify({"insight": output})

    except requests.exceptions.Timeout:
        return jsonify({"error": "⏱️ The assistant took too long. Try again."}), 504
    except Exception as e:
        logging.exception("Error while analyzing entry")
        return jsonify({"error": "⚠️ Something unexpected happened. Please try again."}), 500

@app.route("/prompt", methods=["POST"])
def suggest_prompt():
    try:
        query = (
            "You are a journaling coach. Give the user 1 deep and meaningful self-reflection prompt.\n"
            "Keep it under 20 words. Return just the prompt and nothing else."
        )

        output = ollama_generate(PROMPT_MODEL, query, timeout=TIMEOUT_SECONDS)
        return jsonify({"prompt": output})

    except Exception as e:
        logging.exception("Prompt generation failed")
        return jsonify({"error": "Internal error."}), 500

@app.route("/session/history")
def get_session_history():
    return jsonify(session.get("chat", []))

@app.route("/session/reset", methods=["POST"])
def reset_session():
    session.pop("chat", None)
    return jsonify({"status": "cleared"})

@app.route("/models", methods=["GET"])
def get_available_models():
    """Return available model configuration."""
    return jsonify({
        "generator": GENERATOR_MODEL,
        "fallback": FALLBACK_MODEL,
        "verifier": VERIFIER_MODEL,
        "prompt": PROMPT_MODEL,
        "quality_mode_default": QUALITY_MODE_DEFAULT,
        "retrieval_enabled": RETRIEVAL_ENABLED
    })

if __name__ == "__main__":
    app.run(debug=True)
