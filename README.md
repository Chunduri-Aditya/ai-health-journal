# 🧠 AI Health Journal

**Privacy-first local LLM journaling assistant using Flask + Ollama. All processing happens on your machine—no data leaves your device.**

[![Tests](https://github.com/Chunduri-Aditya/ai-health-journal/actions/workflows/tests.yml/badge.svg)](https://github.com/Chunduri-Aditya/ai-health-journal/actions/workflows/tests.yml)

## What It Does

- **AI-powered insights**: Analyze journal entries with local LLMs (Phi-3, Mistral) via Ollama
- **Multi-model quality pipeline**: Draft → Verify → Revise workflow reduces hallucinations
- **RAG pipeline**: Local vector store (Chroma) for context retrieval from past entries
- **Therapeutic reflections**: Get emotional intelligence insights on your thoughts and patterns
- **Session persistence**: History syncs between frontend and Flask session, persists across page refreshes
- **Privacy-first**: Zero external API calls, no database, all processing on localhost
- **Modern UI**: Typewriter animations, dark mode, collapsible history sidebar, request cancellation, model selector, quality mode toggle
- **DPO Fine-tuning Pipeline**: Build preference datasets and train LoRA adapters to improve groundedness

## Quickstart

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required models pulled:
  ```bash
  ollama pull phi3:3.8b
  ollama pull samantha-mistral:7b
  ```
- Optional: [Chroma](https://www.trychroma.com/) for RAG (installed via requirements.txt)

### Preflight Checks

Before running the app, verify dependencies:

```bash
# Check Python and pip
python3 --version
python3 -m pip --version

# Check core dependencies
python3 -c "import requests; print('requests ok')"
python3 -c "import flask; print('flask ok')"

# Check optional RAG dependency
python3 -c "import chromadb; print('chromadb ok')" || echo "⚠️  chromadb not available - RAG will be disabled"
```

**Note:** If Chroma import fails (dependency conflicts), RAG will be automatically disabled. The app will function normally without RAG.

### Installation

```bash
git clone https://github.com/Chunduri-Aditya/ai-health-journal.git
cd ai-health-journal
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration (Optional)

Create a `.env` file to customize models and features:

```bash
# Model Configuration
GENERATOR_MODEL=phi3:3.8b
FALLBACK_MODEL=phi3:3.8b
VERIFIER_MODEL=samantha-mistral:7b
PROMPT_MODEL=samantha-mistral:7b

# Feature Flags
QUALITY_MODE_DEFAULT=false
RETRIEVAL_ENABLED=true
GROUNDEDNESS_THRESHOLD=0.75
```

### Run

```bash
# Ensure Ollama is running: ollama serve
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser.

**UI Features:**
- **Model Selector**: Choose generator or prompt model
- **Quality Mode Toggle**: Enable Draft → Verify → Revise pipeline for higher accuracy
- **Fast Mode** (Quality Mode OFF): Single-model generation (backward compatible)

---

## DPO Dataset Builder (Baseline vs Quality) — Reproducible Eval → Pairing → Training

This repo builds a DPO preference dataset by:
1. Generating **baseline** answers (intentionally weaker),
2. Generating **quality** answers (stronger + guardrails),
3. Scoring both on the same rubric,
4. Converting results into **(prompt, chosen, rejected)** pairs,
5. Training a model via **DPO**.

**The key idea:** If baseline and quality behave identically, deltas are ~0 and pairs starve. So baseline must be *meaningfully weaker* and quality must be *meaningfully stronger*.

### What "Final Re-Evaluation" Means

A successful re-evaluation should show:
- **Non-trivial deltas** between baseline and quality (not all ~0.000)
- **Healthy number of preference pairs** produced from the same eval set
- **Quality remains high** on safety/faithfulness while baseline drops on some cases

**Targets:**
- **Pairs >= 20–30** for a tiny experiment on ~50 prompts
- For real training: **500+ pairs** (ideally 2k–10k) by expanding prompt variants

---

## Repo Entry Points

### Core Application
- `app.py` - Main Flask app, defines `baseline_json` and `quality` modes
- `llm_client.py` - LLM wrapper with temperature/model support
- `generator_prompts.py` - Draft generation prompts
- `verifier_prompts.py` - Verification prompts
- `rag_store.py` - Local Chroma vector store

### Evaluation Pipeline
- `evals/run_evals.py` - Runs dataset prompts through chosen mode, writes results JSON
- `evals/build_dpo_dataset.py` - Converts baseline + quality results into DPO pairs JSONL
- `evals/debug_pair_deltas.py` - Prints per-case deltas to diagnose filtering
- `evals/compare_before_after.py` - Compares before/after tuning results

### Datasets
- `evals/quick_tests.jsonl` - 6 quick test cases
- `evals/hard_negatives_hn_v2.jsonl` - 51 hard negative cases (recommended)
- `evals/dataset.jsonl` - 25 test cases

### Training
- `train/train_dpo.py` - DPO LoRA training script
- `train/TRAINING_RUNBOOK.md` - Complete training guide (RTX + Mac)
- `train/dpo_pairs_*.jsonl` - Generated preference pairs

### Results
- `evals/results/` - Output JSON files for baseline and quality runs
- `train/` - DPO pairs JSONL output

---

## Environment Setup

### 1. Create a Clean Venv

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python3 -m pip install -U pip
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Sanity Check (Important if you saw Chroma/numpy weirdness):**

```bash
python3 -c "import numpy; import torch; import sklearn; import pandas; print('✅ All deps OK')"
```

If something fails, pin numpy (many stacks still prefer <2.0):

```bash
pip install "numpy<2.0"
```

---

## Final Re-Evaluation (The "Truth Test")

### Step A — Run Baseline (Intentionally Weaker)

```bash
python evals/run_evals.py \
  --dataset evals/hard_negatives_hn_v2.jsonl \
  --mode baseline_json
```

**Expected:** Baseline should score 0.80-0.90 on some metrics (intentionally weaker).

### Step B — Run Quality (Stronger Policy)

```bash
python evals/run_evals.py \
  --dataset evals/hard_negatives_hn_v2.jsonl \
  --mode quality
```

**Expected:** Quality should maintain 0.95-1.00 on key metrics.

### Step C — Identify the Exact Files Produced

```bash
BASE_JSON=$(ls -t evals/results/baseline_json_*.json | head -1)
QUAL_JSON=$(ls -t evals/results/quality_*.json | head -1)

echo "BASE_JSON=$BASE_JSON"
echo "QUAL_JSON=$QUAL_JSON"
```

### Step D — Diagnose Deltas

```bash
python evals/debug_pair_deltas.py "$BASE_JSON" "$QUAL_JSON"
```

**You want to see:**
- Some cases with Δ > 0.05 (or comparable lift in your scoring scale)
- Not all ties at 1.0/1.0/1.0

**If deltas are still near zero:**
- Baseline is not actually weaker (prompt still too strict, temp too low, retrieval too strong)
- Dataset is too easy/saturated (add more "hard negative" prompts/variants)
- Evaluator scoring is too coarse or doesn't reward the improvements you care about

### Step E — Build DPO Pairs

```bash
python evals/build_dpo_dataset.py \
  --baseline_json "$BASE_JSON" \
  --quality "$QUAL_JSON" \
  --output train/dpo_pairs_hn_v2.jsonl
```

### Step F — Check How Many Pairs You Got

```bash
wc -l train/dpo_pairs_hn_v2.jsonl
```

**Expected after fixes (for ~51 prompts):**
- **Before fixes:** ~8–13 pairs, almost all deltas ~0
- **After fixes:** Typically 20–30+ pairs (varies with dataset hardness)

---

## How Pairing Works (And Why Filtering Happens)

### Core Rule

If quality is better than baseline on key rubric metrics → create a pair:
- `chosen` = quality_answer
- `rejected` = baseline_answer

### Why Many Pairs Used to Be Filtered

- Scores were saturated and tied (both ~perfect), so quality wasn't measurably better.
- Result: `quality_not_better_than_baseline`

### What the Fixes Changed

**1. Baseline Intentionally Weaker:**
- Higher temperature (0.3 vs 0.0)
- Weaker prompt guardrails
- Less RAG context (top_k=2 vs 3)
- Fewer retries (3 vs 5)

**2. Tie-Breaker Logic:**
- If metric deltas are ~0 or tied, prefer quality when it:
  - Contains fewer unsupported claims
  - Uses safer refusal phrasing correctly
  - Includes guidance like "consult a professional" where appropriate

This converts ties into usable preference pairs without lowering safety filters.

---

## Model Swapping (Keep Prompts + Fine-Tuning Logic Intact)

You should be able to swap the underlying model without changing:
- Prompts
- Evaluation rubric
- DPO pairing logic
- Training script

### Where to Swap Models

Look for these in `llm_client.py` / `app.py`:
- `MODEL_NAME`
- `model=...`
- `base_url=...` (if using local inference)
- Environment variables (recommended)

### Recommended Pattern

Use environment variables so you don't edit code every time:

```bash
export GENERATOR_MODEL="your-model-name-here"
export LLM_TEMPERATURE_BASELINE="0.3"
export LLM_TEMPERATURE_QUALITY="0.0"
```

Then your `llm_client.py` reads:
- Baseline uses `LLM_TEMPERATURE_BASELINE` (or default 0.3)
- Quality uses `LLM_TEMPERATURE_QUALITY` (or default 0.0)

**Important:** Keep the prompting policy and evaluation rubric constant. Only the model changes.

### Helper Script

```bash
source tools/set_model_env.sh ai-health-journal-dpo
```

---

## Training (Using Already-Evaluated JSON Files)

You can train from previously evaluated JSON files without re-running evals.

### To Train from Existing Results

**1. Choose your stored results:**

```bash
BASE_JSON="evals/results/baseline_json_YYYYMMDD_HHMMSS.json"
QUAL_JSON="evals/results/quality_YYYYMMDD_HHMMSS.json"
```

**2. Build pairs:**

```bash
python evals/build_dpo_dataset.py \
  --baseline_json "$BASE_JSON" \
  --quality "$QUAL_JSON" \
  --output train/dpo_pairs_from_saved_results.jsonl
```

**3. Train using that JSONL:**

See `train/TRAINING_RUNBOOK.md` for complete training instructions.

**Quick Start (Smoke Test):**

```bash
RUN_TRAINING_SMOKE=1 python train/train_dpo.py \
  --dataset train/dpo_pairs_hn_v2.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --output_dir train/output_adapter/ \
  --max_steps 5 \
  --use_4bit
```

**Full Training (RTX Laptop):**

```bash
python train/train_dpo.py \
  --dataset train/dpo_pairs_hn_v2.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --output_dir train/output_adapter/ \
  --num_epochs 1 \
  --batch_size 2 \
  --use_4bit
```

**Note:** Training is optional. The system works well without fine-tuning. See `train/TRAINING_RUNBOOK.md` for RTX laptop and Mac (CPU fallback) instructions.

---

## "Clean Folder" Plan (Remove Extras, Keep Only Required)

You want a minimal, clean folder that:
- Contains only the code required to reproduce eval → pairing
- Preserves the evaluated JSON files used for training
- Keeps training artifacts separate

### A) What to KEEP (Minimal Core)

**Code:**
- `app.py`
- `llm_client.py`
- `generator_prompts.py`
- `verifier_prompts.py`
- `rag_store.py`
- `evals/run_evals.py`
- `evals/build_dpo_dataset.py`
- `evals/debug_pair_deltas.py`
- `evals/compare_before_after.py`

**Prompts / Configs:**
- Prompt files (already in code)
- `.env.example` (if present)

**Data:**
- `evals/quick_tests.jsonl` (input prompts)
- `evals/hard_negatives_hn_v2.jsonl` (input prompts)
- `evals/results/` (KEEP your baseline + quality JSON runs)
- `train/` (DPO pairs JSONL output)

**Docs:**
- `README.md` (this file)
- `LICENSE` (recommended)
- `.gitignore` (recommended)
- `docs/ollama_adapter.md` (for training)
- `train/TRAINING_RUNBOOK.md` (for training)

**Tests:**
- `tests/` (for verification)

### B) What to ARCHIVE (Not Delete) Unless You're Sure

- Old experiments / scratch notebooks
- Old eval datasets you might want later
- Raw downloads or caches (especially for RAG)
- Temporary debug logs
- Multiple report files (consolidate into README)

### C) Create a Clean Export Folder (Recommended Approach)

This avoids deleting anything. It creates a clean "release" folder.

```bash
rm -rf clean_export
mkdir -p clean_export

# Copy only the essentials
rsync -av --prune-empty-dirs \
  --include "app.py" \
  --include "llm_client.py" \
  --include "generator_prompts.py" \
  --include "verifier_prompts.py" \
  --include "rag_store.py" \
  --include "README.md" \
  --include "LICENSE" \
  --include ".gitignore" \
  --include "requirements.txt" \
  --include "evals/" \
  --include "evals/run_evals.py" \
  --include "evals/build_dpo_dataset.py" \
  --include "evals/debug_pair_deltas.py" \
  --include "evals/compare_before_after.py" \
  --include "evals/quick_tests.jsonl" \
  --include "evals/hard_negatives_hn_v2.jsonl" \
  --include "evals/results/***" \
  --include "train/" \
  --include "train/train_dpo.py" \
  --include "train/TRAINING_RUNBOOK.md" \
  --include "train/dpo_pairs_*.jsonl" \
  --include "docs/ollama_adapter.md" \
  --include "tests/" \
  --include "templates/" \
  --include "static/" \
  --include "tools/" \
  --exclude "*" \
  ./ clean_export/
```

Now `clean_export/` is your minimal reproducible package.

### D) Optional: Shrink Results While Preserving Training Source

If results are huge, keep only:
- The baseline JSON used
- The quality JSON used
- The resulting DPO pairs JSONL
- Plus a small MANIFEST section in README describing which results were used

---

## Did We Miss Anything? (Quality + Reproducibility Checklist)

### 1. Determinism

- ✅ Set a fixed random seed for any sampling components (if applicable)
- ✅ Record model name + temperature + top_k in results metadata
- ✅ Use deterministic temperature settings (0.0 for quality, 0.3 for baseline)

### 2. Evaluator Sanity

If faithfulness scores seem weird/inconsistent:
- Inspect a couple "low faithfulness" cases and confirm the evaluator is penalizing the right thing
- Make sure refusal/uncertainty isn't being treated as "unfaithful"

**Quick Check:**

```python
python3 <<'PY'
import json, glob
b = json.load(open(sorted(glob.glob("evals/results/baseline_json_*.json"))[-1]))
q = json.load(open(sorted(glob.glob("evals/results/quality_*.json"))[-1]))

def cases(obj):
    for k in ["case_results","cases","results","items"]:
        if k in obj: return obj[k]
    raise KeyError(obj.keys())

bc = cases(b); qc = cases(q)

# Check cases with low faithfulness
for i, (cb, cq) in enumerate(zip(bc, qc), 1):
    bf = cb.get('metrics', {}).get('faithfulness', 0)
    qf = cq.get('metrics', {}).get('faithfulness', 0)
    if bf < 0.95 or qf < 0.95:
        print(f"\nCase {i}: base_f={bf:.2f} qual_f={qf:.2f}")
        print("Entry:", cb.get('entry', '')[:100])
PY
```

### 3. Dataset Hardness

If baseline vs quality still tie too often:
- Add more adversarial prompts and variants (ambiguous, trap questions, refusal-needed cases)
- Expand from 51 → 500 prompts quickly by templated variants

### 4. Pair Quality

Before training, spot-check 20 pairs:

```bash
head -20 train/dpo_pairs_hn_v2.jsonl | python3 -c "
import json, sys
for i, line in enumerate(sys.stdin, 1):
    pair = json.loads(line)
    print(f'\n=== Pair {i} ===')
    print('Prompt:', pair['prompt'][:100] + '...')
    print('Chosen:', pair['chosen'][:150] + '...')
    print('Rejected:', pair['rejected'][:150] + '...')
"
```

**Check:**
- `chosen` should be clearly better than `rejected`
- No unsafe "chosen" answers
- `rejected` can be worse, but shouldn't be totally garbage unless that matches your training intent

---

## One-Line Pipeline Recap

**Generate baseline + quality → score both → debug deltas → build DPO pairs → train → iterate.**

Baseline must be weaker, quality must be stronger, and ties should be resolved with a safe tie-breaker so you don't starve training.

---

## Release Gates

Before deploying or merging changes, verify production readiness with these commands:

### 1. Automated Tests
```bash
pytest -q
```
**Expected:** 22 passed, 1 skipped

### 2. Evaluation Harness
```bash
python evals/run_evals.py --dataset evals/quick_tests.jsonl --mode both
```
**Expected:**
- Parse failures: 0 in quality mode
- RAG enabled: true
- Mean faithfulness >= 0.95
- Mean no_invention == 1.00
- Mean answer_relevancy >= 0.95

### 3. DPO Dataset Building (Optional)
```bash
BASE_JSON=$(ls -t evals/results/baseline_json_*.json | head -1)
QUAL=$(ls -t evals/results/quality_*.json | head -1)
python evals/build_dpo_dataset.py --baseline_json "$BASE_JSON" --quality "$QUAL" --output train/dpo_pairs.jsonl
```
**Expected:** Non-empty pairs file created with strict filtering applied

**Full acceptance criteria:** See `docs/ACCEPTANCE_CRITERIA.md`

---

## Architecture

```mermaid
flowchart LR
    UI[Browser UI<br/>index.html + Vanilla JS] -->|POST /analyze| API[Flask app.py<br/>Routes + Session]
    UI -->|POST /prompt| API
    UI -->|GET /session/history| API
    UI -->|POST /session/reset| API
    API -->|POST http://localhost:11434/api/generate| OLLAMA[Ollama Local LLM<br/>phi3:3.8b / samantha-mistral:7b]
    API --> SESS[Flask Session<br/>session['chat'] array]
    API -->|Quality Mode| QUALITY[Draft → Verify → Revise<br/>Multi-Model Pipeline]
    API -->|RAG Enabled| RAG[Chroma Vector Store<br/>Local Context Retrieval]
    
    style UI fill:#e1f5ff
    style API fill:#fff4e1
    style OLLAMA fill:#ffe1f5
    style SESS fill:#e1ffe1
    style QUALITY fill:#f0e1ff
    style RAG fill:#e1fff5
```

---

## API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/` | GET | - | HTML page |
| `/ping` | GET | - | `{"status": "ok"}` |
| `/analyze` | POST | `{"entry": "string", "model": "string" (opt), "quality_mode": bool (opt), "baseline_json_mode": bool (opt)}` | `{"insight": "string", "analysis": {...}}` (quality/baseline_json mode) or `{"insight": "string"}` (fast mode) |
| `/prompt` | POST | - | `{"prompt": "string"}` or `{"error": "string"}` |
| `/session/history` | GET | - | `[{"entry": "string", "response": "string"}, ...]` |
| `/session/reset` | POST | - | `{"status": "cleared"}` |
| `/models` | GET | - | `{"generator": "...", "fallback": "...", "verifier": "...", "prompt": "...", "quality_mode_default": bool, "retrieval_enabled": bool}` |

**Input validation:**
- `/analyze` rejects empty entries (400)
- `/analyze` rejects entries >1000 characters (400)
- All entries are HTML-escaped before processing

**Quality Mode Output Schema:**
When `quality_mode: true` or `baseline_json_mode: true`, `/analyze` returns structured JSON:
```json
{
  "insight": "Formatted text for UI",
  "analysis": {
    "summary": "string",
    "emotions": ["string"],
    "patterns": ["string"],
    "triggers": ["string"],
    "coping_suggestions": ["string"],
    "quotes_from_user": ["string"],
    "confidence": 0.0-1.0
  }
}
```

---

## Testing

See `docs/testing.md` for detailed testing instructions.

**Quick Test:**
```bash
pytest -q
# Expected: 22 passed, 1 skipped
```

**Manual E2E Tests:**
- Flow A (Baseline): Start app, submit entry, refresh, new session
- Flow B (Quality Mode): Toggle quality ON, submit entry, confirm structured JSON
- Flow C (AbortController): Submit entry, click Stop
- Flow D (RAG On/Off): Enable retrieval, submit related entries

---

## Privacy & Security

- ✅ **Local-only**: All processing on localhost (Ollama + same-origin Flask)
- ✅ **No external APIs**: Zero external network calls
- ✅ **No database**: Flask session only (cookie-based, server-side)
- ✅ **No logging of sensitive data**: Journal entries and full model outputs never logged
- ✅ **RAG is local**: Chroma vector store runs entirely on your machine

---

## Benchmarks

Run latency benchmarks:

```bash
python scripts/benchmark.py
```

**Expected:** Mean/median/p95 latency for `/analyze` endpoint.

---

## Development

### Project Structure

```
ai-health-journal/
├── app.py                 # Main Flask application
├── llm_client.py          # Ollama client wrapper
├── generator_prompts.py    # Draft generation prompts
├── verifier_prompts.py     # Verification prompts
├── rag_store.py           # Chroma vector store
├── evals/                 # Evaluation pipeline
│   ├── run_evals.py       # Run evaluations
│   ├── build_dpo_dataset.py  # Build DPO pairs
│   ├── debug_pair_deltas.py # Debug deltas
│   ├── compare_before_after.py # Compare results
│   ├── quick_tests.jsonl  # Quick test dataset
│   └── hard_negatives_hn_v2.jsonl  # Hard negatives dataset
├── train/                 # DPO training
│   ├── train_dpo.py       # Training script
│   └── TRAINING_RUNBOOK.md # Training guide
├── tests/                 # Automated tests
└── docs/                  # Documentation
```

---

## Roadmap

- [x] Multi-model quality pipeline
- [x] RAG integration
- [x] Evaluation harness
- [x] DPO dataset builder
- [x] Baseline JSON mode (weaker policy)
- [ ] Expand to 500+ DPO pairs
- [ ] DPO LoRA training (optional)
- [ ] Ollama adapter integration
- [ ] Streaming responses
- [ ] Response caching

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest -q`
4. Run evals: `python evals/run_evals.py --dataset evals/quick_tests.jsonl --mode both`
5. Submit a pull request

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/)
- LLM inference via [Ollama](https://ollama.ai/)
- Vector store via [Chroma](https://www.trychroma.com/)
- DPO training via [Hugging Face TRL](https://github.com/huggingface/trl)

---

**Status:** ✅ **PRODUCTION READY** (Local deployment)
