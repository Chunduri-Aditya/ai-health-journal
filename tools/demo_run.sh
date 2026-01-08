#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"

echo "== Ping =="
curl -s "$BASE_URL/ping" | jq . || curl -s "$BASE_URL/ping"
echo

ENTRY1="I feel stressed about work and not sleeping well."
ENTRY2="I had a really good day and felt grateful."

echo
echo "== Baseline JSON =="
curl -s -X POST "$BASE_URL/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"entry\":\"$ENTRY1\",\"baseline_json_mode\":true}" | jq '{insight, analysis:{summary,confidence,grounding_sources}}' || echo

echo
echo "== Quality Mode =="
curl -s -X POST "$BASE_URL/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"entry\":\"$ENTRY1\",\"quality_mode\":true}" | jq '{insight, analysis:{summary,confidence,grounding_sources}}' || echo

echo
echo "== LangChain Chain Mode =="
curl -s -X POST "$BASE_URL/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"entry\":\"$ENTRY2\",\"mode\":\"langchain_chain\"}" | jq '{insight, analysis:{summary,confidence,grounding_sources}}' || echo

echo
echo "== Transcribe (if Whisper installed) =="
if [ -f "sample.webm" ]; then
  curl -s -X POST "$BASE_URL/transcribe" \
    -F "audio=@sample.webm" | jq . || echo
else
  echo "No sample.webm found; skipping transcribe demo."
fi

