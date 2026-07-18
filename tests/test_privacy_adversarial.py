"""Adversarial regression tests for privacy/exfiltration claims made in
PRIVACY.md and this session's PRIVACY_MODE=strict redaction work.

Same two-kind structure as test_crisis_gate_adversarial.py: real regressions
(no xfail) for defenses that DO hold today, and xfail(strict=True) tests that
document confirmed gaps so a future fix is caught here rather than silently.
"""

import base64
import zlib
from dataclasses import replace
from unittest import mock

import pytest

import app
from config import load_config
from privacy.redact import redact


def _decode_session_cookie(cookie_value: str) -> str:
    """Decode a Flask session cookie exactly as an attacker with cookie
    access would: no SECRET_KEY needed, only base64 (and possibly zlib).

    Flask's SecureCookieSessionInterface zlib-compresses the payload when
    that's smaller than the raw form, marking it with a leading "." before
    the itsdangerous payload.timestamp.signature structure. Whether a given
    cookie is compressed depends on payload size/content, so both forms must
    be handled or this silently misreads some cookies as empty.
    """
    compressed = cookie_value.startswith(".")
    raw = cookie_value[1:] if compressed else cookie_value
    payload = raw.split(".")[0]
    padded = payload + "=" * (-len(payload) % 4)
    decoded_bytes = base64.urlsafe_b64decode(padded)
    if compressed:
        decoded_bytes = zlib.decompress(decoded_bytes)
    return decoded_bytes.decode("utf-8", errors="replace")


# ── Held: cloud gates never read credentials while closed ──────────────────
class TestCloudGatesFailClosed:
    def test_llm_gate_never_reads_anthropic_key_when_backend_is_ollama(self, monkeypatch):
        from providers.factory import get_llm_provider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-CANARY-SHOULD-NEVER-BE-READ")
        cfg = replace(load_config(), llm_backend="ollama", allow_cloud_llm=False)

        calls = []
        real_get = __import__("os").environ.get

        def spy(key, *a, **kw):
            if key == "ANTHROPIC_API_KEY":
                calls.append(key)
            return real_get(key, *a, **kw)

        with mock.patch("os.environ.get", side_effect=spy):
            provider = get_llm_provider(cfg)

        assert type(provider).__name__ == "OllamaProvider"
        assert calls == []

    def test_llm_gate_never_reads_anthropic_key_when_allow_cloud_llm_false(self, monkeypatch):
        from providers.factory import get_llm_provider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-CANARY-SHOULD-NEVER-BE-READ")
        cfg = replace(load_config(), llm_backend="anthropic", allow_cloud_llm=False)

        calls = []
        real_get = __import__("os").environ.get

        def spy(key, *a, **kw):
            if key == "ANTHROPIC_API_KEY":
                calls.append(key)
            return real_get(key, *a, **kw)

        with mock.patch("os.environ.get", side_effect=spy):
            provider = get_llm_provider(cfg)

        assert type(provider).__name__ == "OllamaProvider"
        assert calls == []

    def test_pinecone_gate_raises_rather_than_silently_connecting(self):
        from vector_store.factory import get_vector_store

        cfg = replace(
            load_config(), retrieval_enabled=True, vector_backend="pinecone",
            allow_cloud_vectorstore=False,
        )
        with mock.patch("vector_store.factory.load_config", return_value=cfg):
            with pytest.raises(RuntimeError, match="cloud_vectorstore_not_enabled"):
                get_vector_store()


# ── Held: redact() correctly scrubs what it claims to scrub ────────────────
class TestRedactionHoldsForDocumentedCategories:
    def test_email_redacted(self):
        assert redact("reach me at jane@example.com") == "reach me at [REDACTED_EMAIL]"

    def test_us_phone_redacted(self):
        assert "555-123-4567" not in redact("call 555-123-4567")

    # FIXED (were xfail in test_redact_covers_common_pii_categories below):
    # privacy/redact.py now covers SSN, credit card (shape-based, not Luhn-
    # validated), IPv4, and common international phone groupings.
    @pytest.mark.parametrize(
        "label,text",
        [
            ("ssn", "my SSN is 123-45-6789"),
            ("credit_card", "card number 4111 1111 1111 1111 exp 12/27"),
            ("intl_phone", "you can reach me at +44 20 7946 0958"),
            ("ip_address", "my home IP is 192.168.1.105"),
        ],
    )
    def test_newly_covered_pii_categories(self, label, text):
        assert redact(text) != text, f"{label} passed through unredacted: {text!r}"


# ── Gap: still not covered, and why (deliberate scope boundary) ────────────
@pytest.mark.xfail(
    reason="deliberately out of scope: street addresses and full names need "
    "semantic understanding (NER / a name-and-address dictionary) a regex "
    "cannot reliably provide -- any attempt would either miss most real "
    "instances or over-redact ordinary capitalized phrases constantly. "
    "See privacy/redact.py's module docstring.",
    strict=True,
)
@pytest.mark.parametrize(
    "label,text",
    [
        ("street_address", "I live at 742 Evergreen Terrace, Springfield, IL 62704"),
        ("full_name", "My name is Aditya Chunduri and I work at Acme Corp"),
    ],
)
def test_redact_does_not_cover_address_or_name(label, text):
    assert redact(text) != text, f"{label} passed through unredacted: {text!r}"


# ── Session cookie: does the code path that populates it actually redact? ──
class TestSessionCookieRedaction:
    """_append_to_session (app.py) is on the /analyze request path. Flask's
    default session is a signed-but-UNENCRYPTED cookie: readable via base64
    by anyone with cookie access (devtools, an extension, XSS, a non-HTTPS
    network hop), no SECRET_KEY needed to decode it, only to forge it. So it
    needs the same PRIVACY_MODE=strict treatment as the RAG store, not a
    separate "it's local disk" trust boundary.

    These drive the REAL /analyze route (legacy, non-quality-mode path,
    provider mocked) rather than writing directly into the session dict --
    that's the only way to test whether the code that populates the cookie
    actually redacts, as opposed to whether a hand-constructed cookie would.
    """

    def test_strict_mode_redacts_entry_in_session_cookie(self, monkeypatch):
        monkeypatch.setattr(app.cfg, "privacy_mode", "strict")
        client = app.app.test_client()
        with mock.patch.object(app._provider, "healthcheck", return_value=True), \
             mock.patch.object(app._provider, "generate", return_value="a generic insight"):
            resp = client.post(
                "/analyze",
                json={"entry": "reach me at jane@example.com about this", "quality_mode": False},
            )
        assert resp.status_code == 200

        cookie = client.get_cookie("session")
        assert cookie is not None
        decoded = _decode_session_cookie(cookie.value)
        assert "jane@example.com" not in decoded
        assert "[REDACTED_EMAIL]" in decoded

    def test_balanced_mode_stores_raw_entry_in_session_cookie(self, monkeypatch):
        """Documents the intentional (not a bug) distinction: balanced is the
        default and preserves prior behavior, matching _store_in_rag's same
        balanced-vs-strict split.
        """
        monkeypatch.setattr(app.cfg, "privacy_mode", "balanced")
        client = app.app.test_client()
        with mock.patch.object(app._provider, "healthcheck", return_value=True), \
             mock.patch.object(app._provider, "generate", return_value="a generic insight"):
            resp = client.post(
                "/analyze",
                json={"entry": "reach me at jane@example.com about this", "quality_mode": False},
            )
        assert resp.status_code == 200

        cookie = client.get_cookie("session")
        decoded = _decode_session_cookie(cookie.value)
        assert "jane@example.com" in decoded

    def test_strict_mode_redacts_ssn_in_session_cookie(self, monkeypatch):
        """FIXED (was xfail): now that redact() covers SSN too, this second
        persistence path (the session cookie) benefits from the same fix as
        the RAG store automatically, since both route through _maybe_redact.
        Full names/addresses remain a real, documented gap (see
        test_redact_does_not_cover_address_or_name) -- the SAME text would
        still leak a name via this cookie even in strict mode.
        """
        monkeypatch.setattr(app.cfg, "privacy_mode", "strict")
        client = app.app.test_client()
        with mock.patch.object(app._provider, "healthcheck", return_value=True), \
             mock.patch.object(app._provider, "generate", return_value="a generic insight"):
            resp = client.post(
                "/analyze",
                json={"entry": "My SSN is 123-45-6789", "quality_mode": False},
            )
        assert resp.status_code == 200

        cookie = client.get_cookie("session")
        decoded = _decode_session_cookie(cookie.value)
        assert "123-45-6789" not in decoded


# ── Critical (config-gated): RAG_NAMESPACE_MODE=user has no authentication ──
class TestUserNamespaceModeHasNoAuthentication:
    """RAG_NAMESPACE_MODE=user is documented (app.py _namespace_for docstring)
    as the mode for 'multi-user deployments'. It resolves the namespace as
    f"user:{request.headers.get(X-User-Id, 'anonymous')}" -- a raw,
    self-asserted client header with no token, cookie, or auth binding of any
    kind. Whoever sends the same header value gets full read/write access to
    that namespace's entire RAG-indexed journal history. This is dormant
    under the default RAG_NAMESPACE_MODE=session (Flask's signed cookie IS a
    real auth boundary), but 'user' mode is a shipped, documented, reachable
    option with zero access control -- exactly in the deployment scenario
    (multi-user) where it matters most.
    """

    @pytest.mark.xfail(
        reason="CRITICAL (config-gated on RAG_NAMESPACE_MODE=user): namespace "
        "is a self-asserted header with no authentication; anyone who sends "
        "the same X-User-Id value gets full access to that user's RAG history",
        strict=True,
    )
    def test_namespace_requires_more_than_a_guessable_header(self):
        patched_cfg = replace(app.cfg, rag_namespace_mode="user", rag_user_id_header="X-User-Id")
        with mock.patch.object(app, "cfg", patched_cfg):
            with app.app.test_request_context(
                headers={"X-User-Id": "victim-real-identity-8f2a"}
            ):
                victim_ns = app._namespace_for()
            with app.app.test_request_context(
                headers={"X-User-Id": "victim-real-identity-8f2a"}
            ):
                attacker_ns = app._namespace_for()

        # A real fix needs SOME per-request secret (session token, signed
        # header, auth middleware) binding identity -- not just a client-
        # supplied string. This asserts the two resolve to different pools,
        # which will only be true once that binding exists.
        assert victim_ns != attacker_ns

    def test_unauthenticated_clients_share_one_pool_by_default(self):
        """Documents a related but non-xfail fact: omitting the header
        entirely pools ALL such clients into the same 'user:anonymous'
        namespace. This one isn't marked xfail because it's arguably
        intentional fallback behavior, not a bypass of an identity check --
        but it means 'anonymous' is a shared, not isolated, namespace.
        """
        patched_cfg = replace(app.cfg, rag_namespace_mode="user", rag_user_id_header="X-User-Id")
        with mock.patch.object(app, "cfg", patched_cfg):
            with app.app.test_request_context():
                client_a = app._namespace_for()
            with app.app.test_request_context():
                client_b = app._namespace_for()
        assert client_a == client_b == "user:anonymous"
