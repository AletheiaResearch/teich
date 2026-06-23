"""Shared verifier-reward plumbing for teich's two reward-labeled data paths.

Both the seed/verifiable-task path (``runner.py`` + ``converter.py``) and the
Harbor bench path (``bench/runner.py``) end up doing the same three things with a
task verifier's outcome: decide where its sidecar lives, persist it, and attach
the reward to the training rows. Keeping that in one place means both paths label
rows and store rewards identically — a ``verification/<stem>.json`` sidecar next
to the trace, a ``passed`` bool, and a numeric ``reward`` (the explicit value when
the verifier gives one, else the binary 1.0/0.0 implied by ``passed``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VERIFICATION_DIR = "verification"


def verification_sidecar_path(traces_dir: Path, stem: str) -> Path:
    """Canonical sidecar location for a trace: ``<traces_dir>/verification/<stem>.json``."""
    return traces_dir / VERIFICATION_DIR / f"{stem}.json"


def write_verification_sidecar(traces_dir: Path, stem: str, payload: dict[str, Any]) -> Path:
    """Persist a verifier outcome to the canonical sidecar path and return it."""
    path = verification_sidecar_path(traces_dir, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _numeric_reward(value: Any) -> float | None:
    """A real number (bools excluded) as float, else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def reward_from_sidecar_data(data: Any) -> tuple[bool | None, float | None]:
    """Extract ``(passed, reward)`` from a verification sidecar's parsed JSON.

    ``passed`` must be a genuine bool (a corrupt ``"passed": "false"`` string is
    ignored, not coerced). ``reward`` is the explicit numeric reward when present,
    otherwise None so callers fall back to the binary reward from ``passed``.
    """
    if not isinstance(data, dict):
        return None, None
    passed = data.get("passed") if isinstance(data.get("passed"), bool) else None
    return passed, _numeric_reward(data.get("reward"))


def apply_reward_to_row(row: dict[str, Any], *, passed: bool | None, reward: float | None) -> None:
    """Attach a verifier reward to a training row, consistently across both paths.

    Sets ``passed`` when known; sets ``reward`` to the explicit numeric value when
    given, otherwise to the binary 1.0/0.0 implied by ``passed``. Leaves the row
    untouched when neither is known.
    """
    if isinstance(passed, bool):
        row["passed"] = passed
    numeric = _numeric_reward(reward)
    if numeric is not None:
        row["reward"] = numeric
    elif isinstance(passed, bool):
        row["reward"] = 1.0 if passed else 0.0
