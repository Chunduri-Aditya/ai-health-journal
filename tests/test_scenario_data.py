"""Guard: every shipped scenario file is structurally valid.

Cheap and fast so multi-turn scenario JSON can't rot silently (out-of-order
turns, duplicate ids, or an expected retrieval that points at a future turn).
"""

import pytest

from tests.support.scenarios import all_scenario_paths, load_scenario, validate_scenario


def test_scenario_dir_is_not_empty():
    assert all_scenario_paths(), "expected at least one scenario in evals/scenarios/"


@pytest.mark.parametrize("path", all_scenario_paths(), ids=lambda p: p.stem)
def test_scenario_is_valid(path):
    validate_scenario(load_scenario(path))


def test_validator_rejects_forward_reference():
    bad = {
        "suite_id": "bad",
        "description": "turn 1 cannot retrieve a turn that comes later",
        "turns": [
            {"turn": 1, "id": "t1", "entry": "first", "expect_retrieves_prior_ids": ["t2"]},
            {"turn": 2, "id": "t2", "entry": "second"},
        ],
    }
    with pytest.raises(AssertionError, match="not an earlier turn"):
        validate_scenario(bad)


def test_validator_rejects_out_of_order_turns():
    bad = {
        "suite_id": "bad",
        "description": "turn numbers must start at 1 and increment",
        "turns": [{"turn": 2, "id": "t1", "entry": "x"}],
    }
    with pytest.raises(AssertionError, match="out-of-order"):
        validate_scenario(bad)
