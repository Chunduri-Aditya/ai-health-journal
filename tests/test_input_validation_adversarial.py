"""Adversarial input-validation tests across all Flask routes.

No LLM required: the provider is mocked (healthcheck) or the probe never
reaches it (validation happens first). Confirmed live against the real Flask
test client, not asserted from reading the code.
"""

from unittest import mock

import pytest

import app as app_module


@pytest.fixture
def client():
    return app_module.app.test_client()


# ── Held: real defenses confirmed via the actual test client ───────────────
class TestHeldValidation:
    def test_empty_entry_rejected_cleanly(self, client):
        resp = client.post("/analyze", json={"entry": ""})
        assert resp.status_code == 400
        assert resp.is_json

    def test_whitespace_only_entry_rejected_cleanly(self, client):
        resp = client.post("/analyze", json={"entry": "   \n\t  "})
        assert resp.status_code == 400
        assert resp.is_json

    def test_entry_at_exactly_max_length_accepted(self, client):
        with mock.patch.object(app_module._provider, "healthcheck", return_value=True), \
             mock.patch.object(app_module._provider, "generate", return_value="ok"):
            resp = client.post("/analyze", json={"entry": "a" * 1000})
        assert resp.status_code == 200

    def test_entry_one_over_max_length_rejected(self, client):
        resp = client.post("/analyze", json={"entry": "a" * 1001})
        assert resp.status_code == 400
        assert resp.is_json

    def test_provider_offline_returns_clean_503_not_a_crash(self, client):
        with mock.patch.object(app_module._provider, "healthcheck", return_value=False):
            resp = client.post("/analyze", json={"entry": "hello there"})
        assert resp.status_code == 503
        assert resp.is_json

    def test_missing_whisper_dependency_returns_clean_501(self, client):
        resp = client.post("/transcribe", data={})
        assert resp.status_code == 501
        assert resp.is_json

    def test_forged_session_cookie_does_not_crash_or_leak(self, client):
        """A garbage (unsigned/invalid) session cookie must not crash the app
        or return another session's data -- Flask's itsdangerous signing
        should just treat it as no session at all.
        """
        client.set_cookie("session", "totally-bogus-not-a-real-signed-cookie-value")
        resp = client.get("/session/history")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_malformed_json_body_rejected_cleanly(self, client):
        resp = client.post("/analyze", data="{not valid json!!", content_type="application/json")
        assert resp.status_code == 400
        assert resp.is_json

    # FIXED (was TestEntryTypeValidationGap, xfail): app.py's /analyze now
    # checks isinstance(data, dict) and isinstance(entry, str) before ever
    # calling .strip(), so non-string `entry` values and non-object JSON
    # bodies return a clean 400 instead of crashing with an unhandled
    # AttributeError. journal_entry = data.get("entry", "").strip() used to
    # run before any type check or try/except.
    @pytest.mark.parametrize(
        "bad_entry",
        [None, 12345, ["a", "b"], {"nested": "x"}],
        ids=["null", "int", "list", "dict"],
    )
    def test_non_string_entry_returns_clean_400(self, client, bad_entry):
        resp = client.post("/analyze", json={"entry": bad_entry})
        assert resp.status_code == 400
        assert resp.is_json

    def test_top_level_json_array_body_returns_clean_400(self, client):
        """A top-level JSON array body makes request.get_json() return a
        list; list.get('entry') used to raise AttributeError before any
        try/except -- same root cause as the non-string entry case above.
        """
        resp = client.post("/analyze", data="[1,2,3]", content_type="application/json")
        assert resp.status_code == 400
        assert resp.is_json

    def test_debug_mode_is_not_hardcoded_true(self):
        """FIXED (was TestDebugModeHardcoded, xfail): app.py's launch path
        (`if __name__ == "__main__": app.run(...)`, executed by both
        `make run` and start.sh -- not a dev-only branch) now reads
        FLASK_DEBUG from the environment, defaulting to False, instead of
        hardcoding debug=True. Without this, any unhandled exception
        returned the full interactive Werkzeug debugger: source code, local
        variable values at every stack frame, and a PIN-gated code-execution
        console.
        """
        import inspect

        import app as app_module_for_source

        source = inspect.getsource(app_module_for_source)
        assert "app.run(debug=True)" not in source
