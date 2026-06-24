"""Drive ``teich generate --mode bench``.

A thin loop over ``cfg.bench.sources``: each source declares a ``type`` (harbor,
swe-bench), resolved to a backend that turns tasks into native traces + rewards; the
shared harvest (``backends.base.harvest``) routes each into passed/failed/borderline and
writes a per-task ``metadata/`` sidecar. Backends are the only thing that differs per type.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backends import BenchTask, base, get_backend

if TYPE_CHECKING:
    from ..config import BenchSource, Config


def run_bench(
    cfg: Config, *, console: Any = None, resume: bool = False, refresh: bool = False
) -> list[Path]:
    """Run every configured bench source through its backend and harvest reward-labeled traces.

    Tasks within a source run through a bounded pool of size ``cfg.max_concurrency`` (default 1);
    each task is isolated (its own container + per-task output files), the resume-skip is checked
    before dispatch, and the harvest runs on the main thread as results complete. Backends honor
    ``cfg.timeout_seconds`` on their own container runs.
    """
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

    max_workers = max(1, int(cfg.max_concurrency))
    written: list[Path] = []
    for source in sources:
        backend = get_backend(source.type)
        backend.require()
        # Source-level errors (bad spec, download failure) abort the run.
        tasks = list(backend.tasks(cfg, source, refresh=refresh))

        pending: list[BenchTask] = []
        for task in tasks:
            if resume:
                existing = base.existing_output(cfg, base.bench_stem(source, task.id))
                if existing is not None:
                    out(f"[yellow]bench: skipping {task.id} (already harvested)[/yellow]")
                    written.append(existing)
                    continue
            pending.append(task)
        out(
            f"[blue]bench[{source.type}]: {source.source} -> {len(pending)} task(s) "
            f"(concurrency {max_workers})[/blue]"
        )
        if not pending:
            continue

        # Bind src + the bound run method as defaults so the closure doesn't capture the
        # loop variables by reference (ruff B023).
        def _run(
            task: BenchTask, src: BenchSource = source, run=backend.run
        ) -> tuple[BenchTask, base.BenchRun]:
            return task, run(cfg, src, task)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    _, run = future.result()
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
