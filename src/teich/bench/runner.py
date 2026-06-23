"""Drive the optional `harbor` package over Harbor-format benchmark tasks.

`teich generate --mode bench` runs each task in its own environment image via
harbor's built-in agent for the configured provider (codex/claude_code/pi/hermes),
then ingests the agent's native session (which harbor exports to the agent
``logs_dir/sessions``) through teich's `converter.py` and attaches the task
verifier's ``reward.json`` — producing teich's normalized, reward-labeled rows.

harbor is an optional dependency (the ``bench`` extra) and is imported lazily so a
plain teich install never needs it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..converter import convert_traces_to_training_data

if TYPE_CHECKING:
    from ..config import Config

HARBOR_INSTALL_HINT = (
    "Bench mode needs the optional 'bench' extra: install with "
    "`pip install 'teich[bench]'` (requires Python 3.12+)."
)

# teich agent provider -> harbor AgentName value.
_PROVIDER_TO_AGENT: dict[str, str] = {
    "codex": "codex",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "claude_code": "claude-code",
    "pi": "pi",
    "hermes": "hermes",
}


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


def _agent_name_for(provider: str) -> str:
    name = _PROVIDER_TO_AGENT.get(provider.strip().lower())
    if not name:
        raise RuntimeError(
            f"Bench mode does not support agent provider {provider!r}; "
            "use one of: codex, claude-code, pi, hermes."
        )
    return name


def _agent_auth_env(cfg: Config) -> dict[str, str]:
    """Model credentials for the in-container agent from teich's `api` config.

    Uses an API key (+ optional base_url for OpenRouter/OpenAI-compatible). The
    Codex ChatGPT-subscription/broker path is intentionally not used here.
    """
    env: dict[str, str] = {}
    api_key = cfg.get_api_key()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
        env["OPENROUTER_API_KEY"] = api_key
    base_url = cfg.get_base_url()
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    return env


def _resolve_task_dirs(source: str) -> list[Path]:
    """Resolve bench.source to one or more Harbor task directories (local only for now)."""
    path = Path(source).expanduser()
    if not path.exists():
        raise RuntimeError(
            f"bench.source not found: {source}. "
            "Local task directories are supported; git/HF sources are not wired yet."
        )
    if (path / "task.toml").is_file():
        return [path]
    tasks = sorted(d for d in path.iterdir() if d.is_dir() and (d / "task.toml").is_file())
    if not tasks:
        raise RuntimeError(f"No Harbor tasks (a task.toml) found under {source}.")
    return tasks


def _build_trial_config(cfg: Config, task_dir: Path, trials_dir: Path) -> Any:
    """Build a harbor TrialConfig from teich config (agent provider, model, api auth, backend)."""
    from harbor.models.agent.name import AgentName
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import TaskConfig, TrialConfig

    config = TrialConfig(task=TaskConfig(path=task_dir), trials_dir=trials_dir)
    config.agent.name = AgentName(_agent_name_for(cfg.get_agent_provider()))
    model = cfg.get_effective_model()
    if model:
        config.agent.model_name = model
    config.agent.env.update(_agent_auth_env(cfg))
    config.environment.type = EnvironmentType(cfg.bench.backend)
    return config


def _reward_from_sidecar(reward_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(reward_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _locate_trial_outputs(trial: Any) -> tuple[Path | None, dict[str, Any] | None]:
    """Best-effort: find the native session dir + reward from a finished harbor trial."""
    paths = getattr(trial, "paths", None)
    agent_dir = getattr(paths, "agent_dir", None)
    base = Path(agent_dir).parent if agent_dir else None
    sessions_dir = Path(agent_dir) / "sessions" if agent_dir else None
    if sessions_dir is not None and not sessions_dir.exists():
        sessions_dir = None
    reward: dict[str, Any] | None = None
    if base is not None:
        for reward_path in sorted(base.rglob("reward.json")):
            reward = _reward_from_sidecar(reward_path)
            if reward is not None:
                break
    return sessions_dir, reward


def _ingest_session_dir(
    cfg: Config,
    sessions_dir: Path,
    reward: dict[str, Any] | None,
    task_name: str,
) -> list[Path]:
    """Convert a native session dir to teich rows, attach reward, and write to output."""
    rows = convert_traces_to_training_data(sessions_dir)
    passed = reward.get("passed") if isinstance(reward, dict) else None
    reward_value = reward.get("reward") if isinstance(reward, dict) else None
    for row in rows:
        if isinstance(passed, bool):
            row["passed"] = passed
        if isinstance(reward_value, (int, float)) and not isinstance(reward_value, bool):
            row["reward"] = float(reward_value)
        elif isinstance(passed, bool):
            row["reward"] = 1.0 if passed else 0.0
    out_path = cfg.output.traces_dir / f"bench-{task_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if reward is not None:
        sidecar = cfg.output.traces_dir / "verification" / f"bench-{task_name}.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(reward, ensure_ascii=False, indent=2), encoding="utf-8")
    return [out_path]


async def _create_and_run(config: Any) -> tuple[Any, Any]:
    from harbor.trial.trial import Trial

    trial = await Trial.create(config)
    result = await trial.run()
    return trial, result


def run_bench(cfg: Config, *, console: Any = None, resume: bool = False) -> list[Path]:
    """Run benchmark tasks from ``cfg.bench.source`` and write reward-labeled traces."""
    source = (cfg.bench.source or "").strip()
    if not source:
        raise RuntimeError(
            "--mode bench requires bench.source in config "
            "(a Harbor task directory, a git repo, or an HF dataset of tasks)."
        )
    _require_harbor()
    task_dirs = _resolve_task_dirs(source)
    trials_dir = cfg.output.traces_dir / "bench-trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task_dir in task_dirs:
        if console is not None:
            console.print(f"[blue]bench: running {task_dir.name}[/blue]")
        config = _build_trial_config(cfg, task_dir, trials_dir)
        trial, _result = asyncio.run(_create_and_run(config))
        sessions_dir, reward = _locate_trial_outputs(trial)
        if sessions_dir is None:
            if console is not None:
                console.print(f"[yellow]bench: no native session captured for {task_dir.name}[/yellow]")
            continue
        written.extend(_ingest_session_dir(cfg, sessions_dir, reward, task_dir.name))
    return written
