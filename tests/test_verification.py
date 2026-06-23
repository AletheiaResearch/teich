"""Tests for the shared verifier-reward plumbing used by both the seed/verifiable
path and the Harbor bench path (sidecar location/write + row reward labeling)."""

import json

from teich.verification import (
    apply_reward_to_row,
    reward_from_sidecar_data,
    verification_sidecar_path,
    write_verification_sidecar,
)


def test_sidecar_path_is_canonical(tmp_path):
    assert verification_sidecar_path(tmp_path, "trace-01") == tmp_path / "verification" / "trace-01.json"


def test_write_verification_sidecar_round_trips(tmp_path):
    payload = {"passed": True, "reward": 1.0, "verifier": "test.sh"}
    path = write_verification_sidecar(tmp_path, "bench-add-bug", payload)
    assert path == tmp_path / "verification" / "bench-add-bug.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_reward_from_sidecar_data():
    assert reward_from_sidecar_data({"passed": True, "reward": 0.5}) == (True, 0.5)
    assert reward_from_sidecar_data({"passed": False}) == (False, None)
    # A corrupt non-bool "passed" must not be coerced; a bool reward is not numeric.
    assert reward_from_sidecar_data({"passed": "false", "reward": True}) == (None, None)
    assert reward_from_sidecar_data("nope") == (None, None)


def test_apply_reward_binary_from_passed():
    row_pass: dict = {}
    apply_reward_to_row(row_pass, passed=True, reward=None)
    assert row_pass == {"passed": True, "reward": 1.0}

    row_fail: dict = {}
    apply_reward_to_row(row_fail, passed=False, reward=None)
    assert row_fail == {"passed": False, "reward": 0.0}


def test_apply_reward_explicit_numeric_wins():
    row: dict = {}
    apply_reward_to_row(row, passed=True, reward=0.5)
    assert row == {"passed": True, "reward": 0.5}


def test_apply_reward_numeric_without_passed():
    row: dict = {}
    apply_reward_to_row(row, passed=None, reward=0.7)
    assert row == {"reward": 0.7}


def test_apply_reward_noop_when_unknown():
    row: dict = {"messages": []}
    apply_reward_to_row(row, passed=None, reward=None)
    assert row == {"messages": []}
    # A bool reward is not a valid numeric reward and is ignored.
    apply_reward_to_row(row, passed=None, reward=True)  # type: ignore[arg-type]
    assert row == {"messages": []}
