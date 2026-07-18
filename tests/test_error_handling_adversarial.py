"""Adversarial tests for error-handling hygiene: does any internal detail
leak to the client on failure?

Fail-closed behavior for the crisis gate and the cloud LLM/vector-store gates
is already covered by test_crisis_gate_adversarial.py and
test_privacy_adversarial.py::TestCloudGatesFailClosed -- not duplicated here.
This file covers what's left: whether failure responses leak implementation
detail, checked directly against the real /analyze error-handling branches.
"""

from unittest import mock

import pytest

import app as app_module


@pytest.fixture
def client():
    return app_module.app.test_client()


class TestErrorResponsesDoNotLeakInternals:
    def test_pipeline_failure_does_not_leak_internal_stage_tag(self, client):
        """FIXED (was xfail): the json_parse_failed branch now logs the
        internal stage tag server-side and returns the same generic message
        as every other branch in this function, instead of returning the raw
        exception string verbatim.
        """
        with mock.patch.object(app_module._provider, "healthcheck", return_value=True), \
             mock.patch.object(
                 app_module, "_run_quality_pipeline",
                 side_effect=ValueError("json_parse_failed:stage=draft"),
             ):
            resp = client.post("/analyze", json={"entry": "hello", "quality_mode": True})
        assert resp.status_code == 500
        body = resp.get_json()
        assert "json_parse_failed" not in body.get("error", "")
        assert "stage=" not in body.get("error", "")

    def test_generic_exception_returns_friendly_message_not_repr(self, client):
        """Held: an unexpected (non-ValueError) exception in the pipeline
        does NOT leak the exception's own message/repr to the client.
        """
        with mock.patch.object(app_module._provider, "healthcheck", return_value=True), \
             mock.patch.object(
                 app_module, "_run_quality_pipeline",
                 side_effect=RuntimeError("super secret internal path: /etc/whatever"),
             ):
            resp = client.post("/analyze", json={"entry": "hello", "quality_mode": True})
        assert resp.status_code == 500
        body = resp.get_json()
        assert "/etc/whatever" not in body.get("error", "")
        assert "super secret" not in body.get("error", "")
