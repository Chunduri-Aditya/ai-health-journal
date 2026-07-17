# Behavior Assets

This directory is not part of the primary `/analyze` request path today.

It stores behavior-oriented assets used by evaluation and distillation workflows:

- `rules.json` - structured behavior rules
- `failure_patterns.json` - recurring failure taxonomy
- `few_shot.jsonl` - examples that can seed future prompt conditioning
- `loader.py` - helper code for loading the above assets

Intended workflow:

1. Run evals and collect artifacts.
2. Use `tools/distill_evals_to_behavior.py` to derive or refresh behavior assets.
3. Feed those assets back into future prompt-engineering or fine-tuning work.

Guidance:

- Treat generated JSON / JSONL files here as derived artifacts unless a file explicitly says otherwise.
- If this folder becomes part of the runtime path later, update both this README and `docs/upgrades/07-cleanup.md`.
- If a file here is no longer used by any tool or doc, either wire it in, test it, or delete it.
