"""Drive ``teich generate --mode bench``.

A thin loop over ``cfg.bench.sources``: each source declares a ``type`` (harbor,
swe-bench), resolved to a backend that turns tasks into native traces + rewards; the
shared harvest (``backends.base.harvest``) routes each into passed/failed/borderline and
writes a per-task ``metadata/`` sidecar. Backends are the only thing that differs per type.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backends import base, get_backend

if TYPE_CHECKING:
    from ..config import Config


def run_bench(
    cfg: Config, *, console: Any = None, resume: bool = False, refresh: bool = False
) -> list[Path]:
    """Run every configured bench source through its backend and harvest reward-labeled traces."""
    sources = cfg.bench.sources
    if not sources:
        raise RuntimeError(
            "--mode bench requires at least one entry in bench.sources, e.g.\n"
            "  bench:\n"
            "    sources:\n"
            "      - { type: harbor, source: terminal-bench@2.0 }\n"
            "      - { type: swe-bench, source: SWE-bench/SWE-bench_Verified }"
        )

    def out(message: str) -> None:
        if console is not None:
            console.print(message)

    written: list[Path] = []
    for source in sources:
        backend = get_backend(source.type)
        backend.require()
        # Source-level errors (bad spec, download failure) abort; a single task's failure skips.
        tasks = list(backend.tasks(cfg, source, refresh=refresh))
        out(f"[blue]bench[{source.type}]: {source.source} -> {len(tasks)} task(s)[/blue]")
        for task in tasks:
            stem = base.bench_stem(source, task.id)
            existing = base.existing_output(cfg, stem)
            if resume and existing is not None:
                out(f"[yellow]bench: skipping {task.id} (already harvested)[/yellow]")
                written.append(existing)
                continue
            out(f"[blue]bench: running {task.id}[/blue]")
            try:
                run = backend.run(cfg, source, task)
            except Exception as exc:  # one task's failure (docker/agent/grade) — skip it
                out(f"[red]bench: {task.id}: failed ({type(exc).__name__}: {exc})[/red]")
                continue
            if not run.native_lines:
                out(f"[yellow]bench: no trace harvested for {task.id}[/yellow]")
                continue
            paths, split = base.harvest(cfg, source, task, run)
            primary = base.primary_score(run.rewards)
            score = f"reward={primary:g}" if primary is not None else "unscored"
            out(f"[green]bench: {task.id}: {split} ({score})[/green]")
            written.extend(paths)
    return written
