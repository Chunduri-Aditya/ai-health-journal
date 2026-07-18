"""Adversarial regression tests for RAG store correctness: self-exclusion,
empty-store behavior, write-failure handling, and schema strictness.

Uses a real (ephemeral, tempdir-backed) ChromaStore where the finding is
about ChromaStore's own behavior -- fakes would just test the fake.
"""

import tempfile
from unittest import mock

import pytest
from pydantic import ValidationError

import app as app_module
from schemas.analysis import AnalysisOutput
from schemas.verifier import VerifierVerdict
from vector_store.chroma_store import ChromaStore


@pytest.fixture
def chroma_store():
    with tempfile.TemporaryDirectory() as tmp:
        import os

        old = __import__("os").environ.get("CHROMA_PERSIST_DIR")
        os.environ["CHROMA_PERSIST_DIR"] = tmp
        try:
            yield ChromaStore(default_namespace="test_ns")
        finally:
            if old is not None:
                os.environ["CHROMA_PERSIST_DIR"] = old
            else:
                os.environ.pop("CHROMA_PERSIST_DIR", None)


# ── Held: boundary conditions that correctly return empty, not crash ───────
class TestHeldRAGBoundaries:
    def test_query_on_never_written_namespace_returns_empty(self, chroma_store):
        hits = chroma_store.query("anything", top_k=3, namespace="never_written_ns")
        assert hits == []

    def test_excluding_every_candidate_returns_empty_not_error(self, chroma_store):
        for i in range(3):
            chroma_store.add_entry(
                entry_id=f"id{i}", text=f"entry number {i} about the same topic",
                namespace="excl_probe",
            )
        with mock.patch.object(app_module, "vector_store", chroma_store):
            hits = app_module._retrieve_hits(
                "entry about the same topic", namespace="excl_probe", top_k=3,
                exclude_ids={"id0", "id1", "id2"},
            )
        assert hits == []

    def test_pipeline_failure_never_writes_a_partial_entry(self):
        """_store_in_rag is only called AFTER _run_quality_pipeline/_run_baseline
        return successfully (see the /analyze route), so a draft/verify
        failure can never leave an orphaned half-written entry in the store.
        This test asserts the ordering property directly via source
        inspection, since it's a call-sequencing guarantee, not a runtime
        value to assert on.
        """
        import inspect

        source = inspect.getsource(app_module.analyze_entry)
        store_call_pos = source.index("_store_in_rag(")
        pipeline_call_positions = [
            source.index(name)
            for name in ("_run_quality_pipeline(", "_run_baseline(")
            if name in source
        ]
        assert pipeline_call_positions, "expected at least one pipeline call in analyze_entry"
        assert all(pos < store_call_pos for pos in pipeline_call_positions)


class TestSchemaStrictness:
    def test_analysis_output_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            AnalysisOutput.model_validate(
                {"summary": "s", "confidence": 0.5, "totally_unexpected_field": "x"}
            )

    def test_verifier_verdict_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            VerifierVerdict.model_validate(
                {"groundedness_score": 0.5, "rewrite_required": False, "sneaky_extra_key": "x"}
            )


# ── Gap: a failed RAG write is completely silent to the caller ─────────────
class TestWriteFailureSignaling:
    """FIXED (was TestSilentWriteFailure, xfail): ChromaStore.add_entry
    (vector_store/chroma_store.py) now returns True/False instead of always
    None, and _store_in_rag propagates that as its own return value plus a
    WARNING log at the app layer. Deliberately NOT a raised exception: the
    LLM analysis already succeeded by the time this runs, and breaking the
    whole /analyze response over a failed *secondary* RAG write would be the
    wrong availability tradeoff. The fix is that a failure is now a checkable
    signal instead of being indistinguishable from success -- not that it's
    fatal. Same interface contract applied consistently across ChromaStore,
    PineconeStore, and NoOpStore (vector_store/base.py).
    """

    def test_store_in_rag_returns_false_when_the_underlying_write_fails(self, chroma_store):
        with mock.patch.object(chroma_store, "_get_collection") as mock_coll:
            mock_coll.return_value.add.side_effect = RuntimeError("disk full")
            with mock.patch.object(app_module, "vector_store", chroma_store):
                # No exception -- availability is preserved -- but the
                # return value now truthfully reports the failure.
                ok = app_module._store_in_rag(
                    "a private journal entry that should be indexed",
                    "insight text", entry_id="e1", namespace="probe",
                )
        assert ok is False

    def test_store_in_rag_returns_true_on_successful_write(self, chroma_store):
        with mock.patch.object(app_module, "vector_store", chroma_store):
            ok = app_module._store_in_rag(
                "a private journal entry that should be indexed",
                "insight text", entry_id="e2", namespace="probe",
            )
        assert ok is True
