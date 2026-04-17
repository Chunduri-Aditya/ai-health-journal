# Upgrade 05 — Replace Fake Pipeline Progress With Real SSE Streaming

## Problem

The UI simulates pipeline progress using timers disconnected from the backend:

```168:169:templates/index.html
      stageTimer = setTimeout(() => setStage("verify"), 6000);
      stageTimer = setTimeout(() => setStage("revise"), 14000);
```

Two separate defects:

1. **Disconnected from reality.** On a fast model the UI still shows "verifying…" for 6 s even after `/analyze` already returned. On a slow model the UI shows "done" while the backend is still generating.
2. **Clobbered cancellation.** Both `setTimeout` handles are assigned to `stageTimer`, so the first reference is dropped. Any `clearTimeout(stageTimer)` cleanup can only cancel the second timer. The "verify" transition can never be cancelled.

Net result: the staged UI is theater that works against trust, especially when the local model is slow.

## Goal

The stage indicator advances only when the backend actually reaches that stage. Cancellation works. Stage durations are measurable from logs and surfaced in a diagnostics panel.

Use **Server-Sent Events (SSE)** — one-way server→client push, trivial to implement on Flask, works over plain HTTP, no additional dependencies.

## Dependencies

- **Benefits from:** `06-tests-schema.md` — typed event envelopes benefit from pydantic validation.
- **Should preferably land after:** `02-sqlite-persistence.md` — persist per-stage timings into the `analyses` table for `/session/history` to render durations, and establish a stable session id before the first streamed byte is sent.
- **Safe to parallelize with:** `04-transcription.md`.

## Plan

### A. New endpoint `/analyze/stream`

Method: `POST`. Response: `text/event-stream`.

Request body identical to `/analyze`. The non-streaming `/analyze` is kept for backward compatibility and for clients that can't consume SSE (curl smoke tests, eval harness).

Important Flask/session constraint: if this track lands before 02, establish any session cookie state before the first `yield`. Once the stream starts, header mutation becomes brittle.

### B. Event schema

All events share:

```json
{
  "event": "<name>",
  "ts": "2026-04-17T09:15:00.123Z",
  "elapsed_ms": 1523,
  "stage_elapsed_ms": 812,
  "data": { ... }
}
```

Event sequence for quality mode:

| Event              | `data` payload                                                      |
|--------------------|---------------------------------------------------------------------|
| `request_started`  | `{entry_length, model, quality_mode, baseline_json_mode, session_id}` |
| `retrieval_started`| `{top_k, namespace}`                                                |
| `retrieval_completed` | `{hit_count, source_ids}`                                        |
| `draft_started`    | `{model}`                                                           |
| `draft_completed`  | `{chars, attempts}`                                                 |
| `verify_started`   | `{verifier_model}`                                                  |
| `verify_completed` | `{groundedness_score, unsupported_claim_count, safety_flag_count, rewrite_required}` |
| `revise_started`   | `{fallback_model}`                                                  |
| `revise_completed` | `{chars}`                                                           |
| `revise_skipped`   | `{reason}` where reason ∈ {`groundedness_ok`, `verifier_failed`}    |
| `persist_started`  | `{}`                                                                |
| `persist_completed`| `{analysis_id}`                                                     |
| `finished`         | `{insight, analysis, sources, analysis_id, total_ms}`               |
| `error`            | `{code, message, stage?}`                                           |

Baseline JSON mode omits `verify_*`, `revise_*`, `revise_skipped`. Legacy mode omits retrieval/verify/revise entirely and sends only `request_started` → `draft_started` → `draft_completed` → `persist_*` → `finished`.

### C. Server implementation

New module `streaming/sse.py`:

```python
import json, time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

def sse_format(event: str, data: dict, *, event_id: Optional[str] = None) -> str:
    payload = json.dumps(data, default=str)
    lines = [f"event: {event}"]
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {payload}")
    lines.append("")  # blank line ends the event
    return "\n".join(lines) + "\n"

class StageTimer:
    def __init__(self):
        self.t0 = time.monotonic()
        self.stage_t0 = self.t0
    def mark_stage(self):
        self.stage_t0 = time.monotonic()
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)
    def stage_ms(self) -> int:
        return int((time.monotonic() - self.stage_t0) * 1000)
```

`app.py` `/analyze/stream` handler uses a generator:

```python
@app.route("/analyze/stream", methods=["POST"])
def analyze_stream():
    data = request.get_json(silent=True) or {}
    # ... same validation as /analyze ...
    sid = _ensure_session_id()  # do this before the first yielded byte

    def gen():
        timer = StageTimer()
        try:
            yield sse_format("request_started", _envelope(timer, {...}))
            # retrieval
            timer.mark_stage()
            yield sse_format("retrieval_started", _envelope(timer, {...}))
            sources = _retrieve_context(...)
            yield sse_format("retrieval_completed", _envelope(timer, {"hit_count": len(sources), ...}))
            # draft
            timer.mark_stage()
            yield sse_format("draft_started", _envelope(timer, {...}))
            draft = json_generate(...)
            yield sse_format("draft_completed", _envelope(timer, {...}))
            # verify / revise / persist / finished ...
        except Exception as e:
            logging.exception("stream error")
            yield sse_format("error", _envelope(timer, {"code": type(e).__name__, "message": "…"}))

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # disables proxy buffering
            "Connection": "keep-alive",
        },
    )
```

Where `_envelope(timer, data)` returns `{ts, elapsed_ms, stage_elapsed_ms, data}`.

### D. Refactor shared pipeline

Move pipeline orchestration out of the monolithic `_run_quality_pipeline` into smaller steppable functions returning typed results, so both `/analyze` and `/analyze/stream` can consume them:

```
pipeline/
  __init__.py
  retrieval_step.py
  draft_step.py
  verify_step.py
  revise_step.py
```

Each step takes a context dict + returns a `StepResult` (pydantic). Timing is measured at the call site in the handler, not inside the step.

This also makes 06's contract tests much easier — one step can be tested in isolation.

### E. Frontend changes

**`templates/index.html`**

- Remove the `setTimeout` fake-progress block.
- Switch to a Fetch + `ReadableStream` SSE reader (native browser support via `EventSource` only handles GET; since we POST a body, use `fetch` with manual parse; or switch body to JSON query string via POST fetch):

```javascript
async function analyzeStream(entry) {
  const ac = new AbortController();
  currentAbort = ac;
  const resp = await fetch("/analyze/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({entry, quality_mode: qualityMode}),
    signal: ac.signal,
  });
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      handleSseEvent(parseSse(raw));
    }
  }
}
```

- `setStage("retrieving" | "drafting" | "verifying" | "revising" | "done" | "error")` is called from `handleSseEvent`, not a timer.
- Cancellation: `currentAbort.abort()` cleanly terminates the stream.
- UI cancellation should be treated as a client-side abort first; it does not automatically guarantee Ollama generation is cancelled server-side unless a later backend cancellation path is added.
- Show `stage_elapsed_ms` next to each stage label for transparency.

### F. Error semantics

- On mid-stream error, emit an `error` event then **close the stream** (don't send `finished`).
- Client treats an `error` event as terminal and keeps the partial state visible for user action (retry).

### G. Timeout handling

- Stream has a hard ceiling at `TIMEOUT_QUALITY` × 3 (270 s today).
- Client can abort at any time via `AbortController`.
- A client abort should stop UI updates immediately even if the server is still unwinding a model call.

## New / changed interfaces

### `POST /analyze/stream` (new)

Request: identical to `/analyze`.
Response: `text/event-stream` with the schema in section B.

### `/analyze` (unchanged)

Remains synchronous, for CLI/eval clients. Internally calls the same `pipeline/*` steps.

### Config

| Env var                  | Default | Purpose                                  |
|--------------------------|---------|------------------------------------------|
| `STREAM_HEARTBEAT_MS`    | `15000` | Send a `: ping` comment line this often to keep proxies happy |
| `STREAM_MAX_SECONDS`     | `300`   | Hard ceiling for a single stream         |

## Acceptance criteria

1. Opening DevTools → Network, the `/analyze/stream` request shows content type `text/event-stream` and events stream in progressively.
2. Stage indicator advances strictly in response to backend events. Artificially inject a 10 s sleep in `draft_step.py`; UI shows "drafting" for that full 10 s and does not flip to "verifying" early.
3. Clicking "Cancel" during a request aborts the stream and resets UI within 500 ms. If backend model cancellation is not yet wired through, document that server compute may continue briefly after the client disconnects.
4. An injected error in `verify_step.py` emits a single `error` event; client surfaces the message without crashing.
5. `analyses.rewrite_applied` (from 02) is populated correctly using the `revise_started` / `revise_skipped` signal.
6. Legacy `/analyze` still returns the same JSON shape as before (regression contract covered by 06).
7. No remaining `setTimeout` references in `templates/index.html` related to stage progression.

## Risks & open questions

- **Flask dev server + SSE.** Werkzeug's dev server handles SSE but requires `threaded=True` (default in Flask 2+). Double-check in code.
- **Proxy buffering.** Behind nginx or similar, SSE needs `X-Accel-Buffering: no` (handled above) and `proxy_buffering off`. Document for anyone reverse-proxying.
- **Gunicorn.** Sync workers block on streams. Document `gunicorn -k gevent` or switch to `gthread` with enough threads if/when a real WSGI server is introduced.
- **Back-pressure.** The generator `yield`s synchronously; if the client is slow, the server blocks. Acceptable for single-user local use.
- **Client abort semantics.** Fetch cancellation closes the browser-side stream but does not magically cancel upstream model inference. If this becomes materially expensive, add an explicit cancel token between Flask and the Ollama request layer.
- **Event-ID replay.** Out of scope; we're not building `Last-Event-ID` resume today.
- **Fetch streaming in older browsers.** Safari prior to 15 has quirks around `ReadableStream` + POST. The app already targets modern browsers for audio recording; document the requirement.

## Touch list

- `app.py` — new `/analyze/stream` route; keep `/analyze` as-is; share step functions.
- `pipeline/__init__.py` — new.
- `pipeline/retrieval_step.py`, `draft_step.py`, `verify_step.py`, `revise_step.py` — extracted from `_run_quality_pipeline`.
- `streaming/sse.py` — new.
- `streaming/events.py` — event-name constants + pydantic envelope models (under 06).
- `templates/index.html` — swap timer-based stage logic for SSE handler; remove `stageTimer`.
- `static/style.css` — optional: add per-stage-timing label styles.
- `tests/streaming/` — SSE parse + event ordering tests (under 06).
- `README.md` — "Streaming pipeline" section + reverse-proxy notes.
