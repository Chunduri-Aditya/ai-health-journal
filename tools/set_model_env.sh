#!/bin/bash
# Helper script to set model environment variables for Ollama

# Usage: source tools/set_model_env.sh <model_name>
# Example: source tools/set_model_env.sh ai-health-journal-dpo

if [ -z "$1" ]; then
    echo "Usage: source tools/set_model_env.sh <model_name>"
    echo "Example: source tools/set_model_env.sh ai-health-journal-dpo"
    return 1
fi

MODEL_NAME=$1

export GENERATOR_MODEL="$MODEL_NAME"
export FALLBACK_MODEL="$MODEL_NAME"
export VERIFIER_MODEL="samantha-mistral:7b"
export PROMPT_MODEL="samantha-mistral:7b"

echo "✅ Model environment set:"
echo "  GENERATOR_MODEL=$GENERATOR_MODEL"
echo "  FALLBACK_MODEL=$FALLBACK_MODEL"
echo "  VERIFIER_MODEL=$VERIFIER_MODEL"
echo "  PROMPT_MODEL=$PROMPT_MODEL"
echo ""
echo "To persist, add to .env file:"
echo "GENERATOR_MODEL=$MODEL_NAME"
echo "FALLBACK_MODEL=$MODEL_NAME"
