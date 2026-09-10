# File Reorganization Mapping

This document maps old file paths to new paths after the project reorganization.

## Summary

All Python source code has been moved into the `src/` directory for better organization.

## Directory Structure (Before → After)

```
OLD STRUCTURE:
/
├── app.py
├── config.py  
├── llm_client.py
├── model_selection.py
├── journal_gate.py
├── valence.py
├── system_profile.py
├── generator_prompts.py
├── verifier_prompts.py
├── version.py
├── agent/
├── providers/
├── safety/
├── vector_store/
├── schemas/
├── privacy/
├── behavior/
├── service/
├── templates/
└── static/

NEW STRUCTURE:
/
├── src/
│   ├── app.py ✨
│   ├── config.py ✨
│   ├── llm_client.py ✨
│   ├── model_selection.py ✨
│   ├── journal_gate.py ✨
│   ├── valence.py ✨
│   ├── system_profile.py ✨
│   ├── generator_prompts.py ✨
│   ├── verifier_prompts.py ✨
│   ├── version.py ✨
│   ├── agent/
│   ├── providers/
│   ├── safety/
│   ├── vector_store/
│   ├── schemas/
│   ├── privacy/
│   ├── behavior/
│   ├── service/
│   └── web/
│       ├── templates/
│       └── static/
├── tests/ (unchanged location, imports updated)
├── evals/ (unchanged location, imports updated)
├── scripts/ (unchanged location, imports updated)
├── tools/ (unchanged location, imports updated)
├── train/ (unchanged location, imports updated)
├── docs/ (unchanged)
├── storage/ (unchanged)
└── archive/ (unchanged)
```

## File-by-File Mapping

### Core Application Files

| Old Path | New Path |
|----------|----------|
| `/app.py` | `/src/app.py` |
| `/config.py` | `/src/config.py` |
| `/llm_client.py` | `/src/llm_client.py` |
| `/model_selection.py` | `/src/model_selection.py` |
| `/journal_gate.py` | `/src/journal_gate.py` |
| `/valence.py` | `/src/valence.py` |
| `/system_profile.py` | `/src/system_profile.py` |
| `/generator_prompts.py` | `/src/generator_prompts.py` |
| `/verifier_prompts.py` | `/src/verifier_prompts.py` |
| `/version.py` | `/src/version.py` |

### Module Directories

| Old Path | New Path |
|----------|----------|
| `/agent/` | `/src/agent/` |
| `/providers/` | `/src/providers/` |
| `/safety/` | `/src/safety/` |
| `/vector_store/` | `/src/vector_store/` |
| `/schemas/` | `/src/schemas/` |
| `/privacy/` | `/src/privacy/` |
| `/behavior/` | `/src/behavior/` |
| `/service/` | `/src/service/` |
| `/templates/` | `/src/web/templates/` |
| `/static/` | `/src/web/static/` |

## Import Changes

### Within `src/` (Relative Imports)

Files inside `src/` now use relative imports:

```python
# OLD
from config import load_config
from llm_client import call_llm
from providers.factory import get_llm_provider

# NEW
from .config import load_config
from .llm_client import call_llm
from .providers.factory import get_llm_provider
```

### From `tests/`, `evals/`, `scripts/`, `tools/` (Absolute Imports)

Files outside `src/` use absolute imports with the `src.` prefix:

```python
# OLD
from config import load_config
from llm_client import call_llm
import model_selection

# NEW
from src.config import load_config
from src.llm_client import call_llm
import src.model_selection as model_selection
```

## Running the Application

### Flask App (default)

```bash
# OLD
python app.py
# or
python3 -m app

# NEW
python -m src.app
# or via start script
./start.sh
```

### FastAPI Service

```bash
# OLD
uvicorn service.main:app --host 127.0.0.1 --port 8080

# NEW
uvicorn src.service.main:app --host 127.0.0.1 --port 8080
# or via start script
./start.sh --service
```

## Configuration & Paths

### Template and Static Folders

Flask app now explicitly sets paths:

```python
app = Flask(__name__,
           template_folder=os.path.join(_current_dir, 'web', 'templates'),
           static_folder=os.path.join(_current_dir, 'web', 'static'))
```

### Relative Paths

File paths relative to project root are now resolved from `src/`:

```python
# Before
BENCHMARK_LATEST_JSON = Path("evals/reports/job_market_patient_model_benchmark_latest.json")

# After
_PROJECT_ROOT = Path(__file__).parent.parent  # Go up from src/ to project root
BENCHMARK_LATEST_JSON = _PROJECT_ROOT / "evals/reports/job_market_patient_model_benchmark_latest.json"
```

## Testing

All imports have been updated and tested. To verify:

```bash
# Preflight check
./start.sh --check

# Test imports
python -c "from src import app; print('✓ Imports working')"

# Run tests
pytest tests/
```

## Files NOT Moved

These files remain in the root directory:

- **Entry Points**: `start.sh`, `Makefile`
- **Configuration**: `.env.example`, `pytest.ini`, `docker-compose.yml`, `Dockerfile*`, `fly.toml`
- **Dependencies**: `requirements-*.txt`
- **Documentation**: `README.md`, `LICENSE`, `PRIVACY.md`, `*.md`
- **Git**: `.git/`, `.gitignore`, `.github/`
- **IDE**: `.cursor/`, `.impeccable/`
- **Data**: `storage/`, `venv/`
- **Tests**: `tests/`, `evals/`
- **Tools**: `scripts/`, `tools/`, `train/`
- **Docs**: `docs/`
- **Archive**: `archive/`

## Benefits of This Organization

1. **Cleaner Root**: Only 11 configuration/documentation files in root (vs 20+ before)
2. **Clear Structure**: All source code is in `src/`
3. **Logical Grouping**: Web assets in `src/web/`, modules in their own folders
4. **Standard Python**: Follows Python package conventions
5. **Import Clarity**: Clear distinction between internal (relative) and external (absolute) imports
6. **GitHub Ready**: Standard structure recognized by GitHub and IDEs

## Validation

✅ All imports updated (28 files modified)
✅ Flask app tested and working  
✅ Template/static paths verified
✅ File paths relative to project root fixed
✅ Start script updated (`./start.sh`)
✅ Service endpoint updated (FastAPI)

---

**Last Updated**: September 9, 2026  
**Migration Completed**: ✅ All files relocated and tested
