# === Enhanced app.py ===
# Supports journaling prompt generation, chat history display, session tracking

import html
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.urandom(24)

# === CONFIG ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_MODEL = "phi3:3.8b"
PROMPT_MODEL = "samantha-mistral:7b"
MAX_LENGTH = 1000
TIMEOUT_SECONDS = int(os.getenv("AIHJ_OLLAMA_TIMEOUT", "30"))
BENCHMARK_LATEST_JSON = Path("evals/reports/job_market_patient_model_benchmark_latest.json")

logging.basicConfig(level=logging.INFO)

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


def _ollama_models():
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    response.raise_for_status()
    models = []
    for model in response.json().get("models", []):
        name = model.get("name", "")
        if "embed" in name.lower():
            continue
        models.append(
            {
                "name": name,
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
                "digest": model.get("digest"),
            }
        )
    return sorted(models, key=lambda item: item["name"])


@app.route("/models")
def list_models():
    try:
        models = _ollama_models()
        ok = True
        error = None
    except requests.exceptions.RequestException as exc:
        logging.warning("Could not list Ollama models: %s", exc)
        models = [{"name": DEFAULT_MODEL, "size": None, "modified_at": None, "digest": None}]
        ok = False
        error = "Ollama is not reachable."

    return jsonify(
        {
            "ok": ok,
            "error": error,
            "default_model": DEFAULT_MODEL,
            "prompt_model": PROMPT_MODEL,
            "models": models,
        }
    )


@app.route("/benchmark/latest")
def latest_benchmark():
    if not BENCHMARK_LATEST_JSON.exists():
        return jsonify({"ok": False, "error": "No model benchmark has been run yet."}), 404

    try:
        with BENCHMARK_LATEST_JSON.open("r", encoding="utf-8") as handle:
            return jsonify({"ok": True, "benchmark": json.load(handle)})
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Could not read benchmark file: %s", exc)
        return jsonify({"ok": False, "error": "Benchmark report could not be read."}), 500


@app.route("/analyze", methods=["POST"])
def analyze_entry():
    data = request.get_json(silent=True) or {}
    journal_entry = data.get("entry", "").strip()
    model = data.get("model", DEFAULT_MODEL)

    if not journal_entry:
        return jsonify({"error": "Please enter some thoughts before submitting."}), 400
    if len(journal_entry) > MAX_LENGTH:
        return jsonify({"error": f"Entry too long. Limit to {MAX_LENGTH} characters."}), 400

    safe_entry = html.escape(journal_entry)

    try:
        requests.get(OLLAMA_BASE_URL, timeout=2)
    except requests.exceptions.RequestException:
        return jsonify({"error": "The AI assistant is offline. Please try again later."}), 503

    prompt = (
        "You are an emotionally intelligent journaling assistant.\n"
        "When I share a journal entry, analyze it like a therapist.\n"
        "Be concise, warm, and clear.\n"
        "Summarize the core avoided emotion and what it's trying to teach me.\n"
        "Give 2-3 short reflective insights that help me understand myself better.\n"
        "Use gentle, human language. Avoid rambling or restating obvious things.\n\n"
        "If the entry mentions self-harm, disappearing, being better off gone, or not wanting to live, "
        "include immediate support guidance: contact a trusted person now, call or text 988 in the U.S., "
        "or contact emergency services if there is immediate danger. Do not diagnose the user.\n\n"
        f"My journal entry:\n\"{safe_entry}\""
    )

    try:
        started_at = time.perf_counter()
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT_SECONDS
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

        if response.status_code != 200:
            logging.warning(f"LLM error {response.status_code}: {response.text}")
            return jsonify({"error": "The assistant had trouble responding. Please try again."}), 500

        output = response.json().get("response", "").strip()
        for phrase in ["I'm here for you", "Thank you for sharing"]:
            output = output.replace(phrase, "").strip()

        # Save to session history
        if "chat" not in session:
            session["chat"] = []
        session["chat"].append(
            {
                "entry": journal_entry,
                "response": output,
                "model": model,
                "latency_ms": latency_ms,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        )

        return jsonify({"insight": output, "model": model, "latency_ms": latency_ms})

    except requests.exceptions.Timeout:
        return jsonify({"error": "The assistant took too long. Try again."}), 504
    except Exception as e:
        logging.exception("Error while analyzing entry")
        return jsonify({"error": "Something unexpected happened. Please try again."}), 500

@app.route("/prompt", methods=["POST"])
def suggest_prompt():
    try:
        query = (
            "You are a journaling coach. Give the user 1 deep and meaningful self-reflection prompt.\n"
            "Keep it under 20 words. Return just the prompt and nothing else."
        )

        response = requests.post(
            OLLAMA_API_URL,
            json={"model": PROMPT_MODEL, "prompt": query, "stream": False},
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code != 200:
            return jsonify({"error": "Could not fetch prompt."}), 500

        prompt = response.json().get("response", "").strip()
        return jsonify({"prompt": prompt})

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

if __name__ == "__main__":
    app.run(debug=True)
