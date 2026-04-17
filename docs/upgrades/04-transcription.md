# Upgrade 04 — Repair and Modernize Voice Transcription

## Problem

The install and runtime stories don't match:

```184:190:app.py
    try:
        import whisper  # type: ignore
        import tempfile, io
    except ImportError:
        return jsonify({
            "error": "Transcription requires openai-whisper. Install with: pip install openai-whisper"
        }), 501
```

```17:19:requirements-optional.txt
# Whisper voice transcription
# Note: pulls in ctranslate2, av, etc.
faster-whisper>=1.0.0
```

- `requirements-optional.txt` installs **`faster-whisper`** (the `faster_whisper` module).
- `app.py` imports **`whisper`** (the `openai-whisper` module, a different package).
- The 501 error tells users to `pip install openai-whisper`, which is *yet another* heavier dependency (pulls in `torch`, `ffmpeg-python`, etc).

Following `make setup-full` installs `faster-whisper`, then the first transcription request raises `ImportError`, returns 501, points at the wrong package — a broken happy path.

Additional issues:

- `whisper.load_model(...)` is called on **every request** inside the handler → seconds of CPU overhead per call, heavy RAM spike.
- No mime-type or duration validation before decoding.
- Response contains only `text` and `language` — no duration, no per-segment timing, no confidence.

## Goal

A single, working transcription backend (`faster-whisper`) that:

- Installs cleanly via `make setup-full`.
- Loads the model **once** at first use and caches it for the process lifetime.
- Validates input upload before attempting decode.
- Returns `{text, language, duration_s, segments?, average_logprob?}`.
- Documents CPU vs Apple Silicon (Metal via `ct2` CPU fallback) behavior.

## Dependencies

- **None.** Isolated from all other upgrades. Can land anytime.
- Nice-to-have: `06-tests-schema.md` for handler tests (mocked `WhisperModel`).

## Plan

### A. Pick `faster-whisper`

Rationale:
- Already declared in `requirements-optional.txt`.
- Uses `ctranslate2` — 2–4× faster than `openai-whisper` on CPU, smaller memory footprint.
- No `ffmpeg` Python binding required for basic use (`faster-whisper` handles decoding internally via `av`).
- Better fit for "local-first, runs on your laptop" positioning.

### B. New module `transcription/whisper_service.py`

```python
from __future__ import annotations
import logging, os, tempfile, threading
from dataclasses import dataclass
from typing import List, Optional

_model = None
_model_lock = threading.Lock()

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float

@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    duration_s: float
    segments: List[Segment]
    average_logprob: float

def _load_model():
    from faster_whisper import WhisperModel
    model_name = os.getenv("WHISPER_MODEL", "base")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 is CPU-friendly default
    device = os.getenv("WHISPER_DEVICE", "auto")
    logging.info("Loading WhisperModel: model=%s device=%s compute_type=%s", model_name, device, compute_type)
    return WhisperModel(model_name, device=device, compute_type=compute_type)

def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _model = _load_model()
    return _model

def transcribe(audio_bytes: bytes, *, suffix: str = ".webm") -> TranscriptionResult:
    model = get_model()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name
        segments_iter, info = model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
        )
        segs = [
            Segment(start=s.start, end=s.end, text=s.text, avg_logprob=s.avg_logprob)
            for s in segments_iter
        ]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    text = "".join(s.text for s in segs).strip()
    avg = sum(s.avg_logprob for s in segs) / len(segs) if segs else 0.0
    return TranscriptionResult(
        text=text,
        language=info.language,
        duration_s=info.duration,
        segments=segs,
        average_logprob=avg,
    )
```

Notes:
- `_load_model()` is the only place that imports `faster_whisper`, so the app can start even when the package isn't installed.
- `compute_type=int8` is the right CPU default; override to `float16` for GPU.
- `vad_filter=True` helps with ambient silence common in voice journaling.
- Single-shot `threading.Lock` prevents two simultaneous first-requests from loading two copies.
- The explicit cleanup pattern avoids tempfile reopen issues on platforms that dislike `NamedTemporaryFile(delete=True)` with library reopens by path.

### C. `/transcribe` route changes

**`app.py`**

```python
from transcription.whisper_service import get_model, transcribe  # lazy inside the handler

ALLOWED_AUDIO_MIMES = {
    "audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4",
    "audio/x-m4a", "audio/wav", "audio/x-wav",
}

_EXT_BY_MIME = {
    "audio/webm": ".webm", "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
    "audio/wav": ".wav", "audio/x-wav": ".wav",
}

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        import faster_whisper  # type: ignore  # noqa: F401
    except ImportError:
        return jsonify({
            "error": "Transcription requires faster-whisper. Install with: pip install -r requirements-optional.txt"
        }), 501

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided."}), 400

    mime = (audio_file.mimetype or "").lower()
    if mime not in ALLOWED_AUDIO_MIMES:
        return jsonify({"error": f"Unsupported audio mime type: {mime}"}), 415

    audio_bytes = audio_file.read()
    if len(audio_bytes) > cfg.whisper_max_audio_bytes:
        return jsonify({"error": "Audio file too large."}), 413
    if len(audio_bytes) < 1024:
        return jsonify({"error": "Audio file too small / empty."}), 400

    try:
        result = transcribe(audio_bytes, suffix=_EXT_BY_MIME.get(mime, ".webm"))
    except Exception:
        logging.exception("Transcription failed")
        return jsonify({"error": "Transcription failed. Please try again."}), 500

    return jsonify({
        "text": result.text,
        "language": result.language,
        "duration_s": round(result.duration_s, 2),
        "average_logprob": round(result.average_logprob, 4),
        "segments": [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
            for s in result.segments
        ],
    })
```

The MIME gate is only an early rejection filter. Decode can still fail later on malformed or mislabeled uploads, so keep a clear failure path even after the MIME check passes.

### D. Warmup option

New env var `WHISPER_WARMUP=1` — on app startup, call `get_model()` in a background thread. Default off; useful for demos to avoid the first-request stall.

### E. README updates

Add a "Voice transcription" subsection:

- Dependency is `faster-whisper`; `openai-whisper` is **not** supported.
- CPU default (`compute_type=int8`) is fast enough for `base` and `small` models.
- Apple Silicon: `faster-whisper` runs via CTranslate2's CPU path (no Metal). For real GPU acceleration, set `WHISPER_DEVICE=cuda` with a supported CUDA install.
- First request after boot pays the model-load cost (~1–4 s for `base` on CPU). Subsequent requests are fast.
- Set `WHISPER_WARMUP=1` to preload at startup.

### F. Delete stale guidance

- Remove references to `openai-whisper` from README and the 501 error message.
- Confirm no other files mention `import whisper` / `pip install openai-whisper`.

## New / changed interfaces

### Env vars

| Env var              | Default     | Purpose                                   |
|----------------------|-------------|-------------------------------------------|
| `WHISPER_MODEL`      | `base`      | `tiny` \| `base` \| `small` \| `medium`   |
| `WHISPER_COMPUTE_TYPE` | `int8`    | `int8` \| `int8_float16` \| `float16`     |
| `WHISPER_DEVICE`     | `auto`      | `auto` \| `cpu` \| `cuda`                 |
| `WHISPER_WARMUP`     | `0`         | Preload model at startup when `1`         |

### `/transcribe` response (new shape)

```json
{
  "text": "Today was rough, I kept spiraling.",
  "language": "en",
  "duration_s": 4.32,
  "average_logprob": -0.2854,
  "segments": [
    {"start": 0.0, "end": 4.32, "text": "Today was rough..."}
  ]
}
```

## Acceptance criteria

1. Fresh `make setup-full` followed by a single audio POST to `/transcribe` returns 200 with the documented payload — no `ImportError`, no `openai-whisper` install step.
2. The model is loaded exactly once: instrument with a log line + test that asserts it's invoked once across three sequential requests.
3. Uploading a non-audio file (e.g. `image/png`) returns 415 before any decode attempt.
4. Uploading a file over `WHISPER_MAX_AUDIO_BYTES` returns 413.
5. The README's "Voice transcription" section documents the chosen backend and the tradeoffs. No remaining mentions of `openai-whisper`.
6. With `WHISPER_WARMUP=1`, the first real request does not pay the model-load latency.

## Risks & open questions

- **`av` native dep.** `faster-whisper` relies on `av` (PyAV, libav). Wheels exist for macOS/Linux/Windows on modern Python versions. On unusual platforms (Alpine, old glibc), the install may need `ffmpeg` system libraries. Document as a known issue.
- **Streaming transcription.** Not in scope. If later required, `faster-whisper`'s `transcribe(..., word_timestamps=True)` + chunked uploads is the path.
- **Memory.** `base` int8 on CPU ≈ 150 MB resident. `small` ≈ 500 MB. `medium` ≈ 1.5 GB. Call this out in README so no one accidentally runs `medium` on a 4 GB box.
- **Offline decode.** `faster-whisper` downloads model weights on first load (from HuggingFace). Document `HF_HUB_OFFLINE=1` + pre-cached weights for strict-offline use.
- **Payload size.** Full segment arrays are useful for diagnostics, but they can bloat responses on long recordings. If that becomes a UX issue, add an `include_segments` flag as a later follow-up rather than complicating v1 now.

## Touch list

- `app.py` — `/transcribe` handler.
- `transcription/__init__.py` — new.
- `transcription/whisper_service.py` — new.
- `requirements-optional.txt` — confirm `faster-whisper>=1.0.0`, keep `python-multipart`.
- `config.py` — add new `whisper_*` fields if needed (device, compute_type, warmup).
- `README.md` — new "Voice transcription" section; prune stale `openai-whisper` references.
- `tests/transcription/` — handler tests with mocked model (under 06).
