"""Drive the optional `harbor` package over Harbor-format benchmark tasks.

`teich generate --mode bench` runs each task in its own environment image via
harbor's built-in agent for the configured provider (codex/claude_code/pi/hermes),
then ingests the agent's native session (which harbor exports to
``logs_dir/sessions``) through teich's `converter.py` and attaches the task
verifier's ``reward.json`` — producing teich's normalized, reward-labeled rows.

harbor is an optional dependency (the ``bench`` extra) and is imported lazily so a
plain teich install never needs it.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import Config

HARBOR_INSTALL_HINT = (
    "Bench mode needs the optional 'bench' extra: install with "
    "`pip install 'teich[bench]'` (requires Python 3.12+)."
)


def _require_harbor() -> Any:
    """Import harbor, with a clear actionable error if it (or Python 3.12+) is missing."""
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Bench mode requires Python 3.12+ "
            f"(current: {sys.version_info.major}.{sys.version_info.minor}). {HARBOR_INSTALL_HINT}"
        )
    try:
        import harbor
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(HARBOR_INSTALL_HINT) from exc
    return harbor


def run_bench(cfg: Config, *, console: Any = None, resume: bool = False) -> list:
    """Run benchmark tasks from ``cfg.bench.source`` and write reward-labeled traces."""
    source = (cfg.bench.source or "").strip()
    if not source:
        raise RuntimeError(
            "--mode bench requires bench.source in config "
            "(a Harbor task directory, a git repo, or an HF dataset of tasks)."
        )
    _require_harbor()
    # The harbor TrialConfig run + native-session/reward ingest land in the next
    # step; this validates the bench config and dependency surface first.
    raise RuntimeError(
        "bench mode: the harbor task driver is not wired yet (next step). "
        f"Validated source={source!r}, backend={cfg.bench.backend!r}."
    )
