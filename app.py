# === Enhanced app.py ===
# Supports journaling prompt generation, chat history display, session tracking

from flask import Flask, request, jsonify, render_template, session
import requests
import logging
import html
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# === CONFIG ===
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3:3.8b"
PROMPT_MODEL = "samantha-mistral:7b"
MAX_LENGTH = 1000
TIMEOUT_SECONDS = 15

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

@app.route("/analyze", methods=["POST"])
def analyze_entry():
    data = request.get_json()
    journal_entry = data.get("entry", "").strip()
    model = data.get("model", DEFAULT_MODEL)

    if not journal_entry:
        return jsonify({"error": "✋ Please enter some thoughts before submitting."}), 400
    if len(journal_entry) > MAX_LENGTH:
        return jsonify({"error": f"📏 Entry too long. Limit to {MAX_LENGTH} characters."}), 400

    safe_entry = html.escape(journal_entry)

    try:
        requests.get("http://localhost:11434", timeout=2)
    except requests.exceptions.RequestException:
        return jsonify({"error": "🛑 The AI assistant is offline. Please try again later."}), 503

    prompt = (
        "You are an emotionally intelligent journaling assistant.\n"
        "When I share a journal entry, analyze it like a therapist.\n"
        "Be concise, warm, and clear.\n"
        "Summarize the core avoided emotion and what it's trying to teach me.\n"
        "Give 2–3 short reflective insights that help me understand myself better.\n"
        "Use gentle, human language. Avoid rambling or restating obvious things.\n\n"
        f"My journal entry:\n\"{safe_entry}\""
    )

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT_SECONDS
        )

        if response.status_code != 200:
            logging.warning(f"LLM error {response.status_code}: {response.text}")
            return jsonify({"error": "⚠️ The assistant had trouble responding. Please try again."}), 500

        output = response.json().get("response", "").strip()
        for phrase in ["I'm here for you", "Thank you for sharing"]:
            output = output.replace(phrase, "").strip()

        # Save to session history
        if "chat" not in session:
            session["chat"] = []
        session["chat"].append({"entry": journal_entry, "response": output})

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
