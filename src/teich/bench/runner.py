"""Drive the optional `harbor` package over Harbor-format benchmark tasks.

`teich generate --mode bench` runs each task in its own environment image via
harbor's built-in agent for the configured provider (codex/claude_code/pi/hermes),
then ingests the agent's native trace through teich's `converter.py` and attaches
the task verifier's reward — producing teich's normalized, reward-labeled rows.

Trace shapes differ per agent: codex/claude-code export a native session dir
(``agent/sessions/*.jsonl``) the converter reads directly; pi runs with
``--no-session`` and emits only its ``--mode json`` event stream (``agent/pi.txt``),
which we normalize into pi session events before converting. The reward comes from
harbor's ``TrialResult.verifier_result`` (falling back to the on-disk reward file).

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
from ..verification import apply_reward_to_row, write_verification_sidecar

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


def _reward_from_text(reward_path: Path) -> dict[str, Any] | None:
    """Parse harbor's single-value reward.txt into our reward dict."""
    try:
        raw = reward_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return {"reward": value, "passed": value > 0}


def _reward_dict_from_value(value: Any) -> dict[str, Any] | None:
    """Wrap a single numeric reward into our ``{reward, passed}`` shape."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return {"reward": float(value), "passed": float(value) > 0}


def _reward_from_result(result: Any) -> dict[str, Any] | None:
    """Read the reward from harbor's ``TrialResult.verifier_result`` (the canonical source).

    harbor reports ``verifier_result = {"rewards": {"reward": <float>, ...}}``; we
    take ``reward`` when present, else the first numeric reward value.
    """
    verifier = getattr(result, "verifier_result", None)
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else getattr(verifier, "rewards", None)
    if not isinstance(rewards, dict):
        return None
    reward = _reward_dict_from_value(rewards.get("reward"))
    if reward is not None:
        return reward
    for value in rewards.values():
        reward = _reward_dict_from_value(value)
        if reward is not None:
            return reward
    return None


def _reward_from_files(base: Path | None) -> dict[str, Any] | None:
    """Fallback: scan a finished trial dir for harbor's reward sidecar/text file."""
    if base is None:
        return None
    for name in ("rewards.json", "reward.json"):
        for reward_path in sorted(base.rglob(name)):
            reward = _reward_from_sidecar(reward_path)
            if reward is not None:
                return reward
    for reward_path in sorted(base.rglob("reward.txt")):
        reward = _reward_from_text(reward_path)
        if reward is not None:
            return reward
    return None


def _agent_dir(trial: Any) -> Path | None:
    paths = getattr(trial, "paths", None)
    agent_dir = getattr(paths, "agent_dir", None)
    return Path(agent_dir) if agent_dir else None


def _pi_stream_to_session_events(pi_txt: Path) -> list[dict[str, Any]]:
    """Normalize pi's ``--mode json`` event stream into pi session events.

    harbor runs pi with ``--no-session``, so it leaves only the streaming log
    (``session``/``message_start``/``message_end``/``tool_execution_*``/``turn_*``,
    plus a leading non-JSON ``Warning:`` line). teich's converter consumes pi
    *session* events (``{"type": "message", "message": {...}}``), so we keep the
    ``session`` header, turn each completed ``message_end`` into a ``message``, and
    synthesize a ``model_change`` from the first assistant message (the stream has
    no such event, so model metadata would otherwise be lost).
    """
    events: list[dict[str, Any]] = []
    model_change_added = False
    for raw in pi_txt.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue  # skip pi's leading "Warning: ..." line
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session":
            events.append(event)
        elif event_type == "message_end":
            message = event.get("message")
            if not isinstance(message, dict) or not message.get("role"):
                continue
            if not model_change_added and message.get("role") == "assistant":
                provider, model_id = message.get("provider"), message.get("model")
                if isinstance(provider, str) and isinstance(model_id, str):
                    events.append({"type": "model_change", "provider": provider, "modelId": model_id})
                    model_change_added = True
            events.append({"type": "message", "message": message})
    return events


def _harvest_trace(
    cfg: Config, trial: Any, reward: dict[str, Any] | None, task_name: str
) -> list[Path]:
    """Convert a finished trial's native agent trace to reward-labeled teich rows."""
    agent_dir = _agent_dir(trial)
    if agent_dir is None or not agent_dir.exists():
        return []
    # codex / claude-code export a native session dir the converter reads directly.
    sessions = agent_dir / "sessions"
    if sessions.is_dir() and any(sessions.glob("*.jsonl")):
        return _ingest_session_dir(cfg, sessions, reward, task_name)
    # pi runs with --no-session: normalize its --mode json stream into a session file.
    pi_txt = agent_dir / "pi.txt"
    if pi_txt.is_file():
        events = _pi_stream_to_session_events(pi_txt)
        if events:
            norm_dir = cfg.output.traces_dir / "bench-sessions" / task_name
            norm_dir.mkdir(parents=True, exist_ok=True)
            (norm_dir / "pi.jsonl").write_text(
                "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
                encoding="utf-8",
            )
            return _ingest_session_dir(cfg, norm_dir, reward, task_name)
    # last resort: any *.jsonl the agent left behind.
    if any(agent_dir.rglob("*.jsonl")):
        return _ingest_session_dir(cfg, agent_dir, reward, task_name)
    return []


def _ingest_session_dir(
    cfg: Config,
    sessions_dir: Path,
    reward: dict[str, Any] | None,
    task_name: str,
) -> list[Path]:
    """Convert a native session dir to teich rows, attach reward, and write to output."""
    rows = convert_traces_to_training_data(sessions_dir)
    if not rows:
        return []
    passed = reward.get("passed") if isinstance(reward, dict) else None
    reward_value = reward.get("reward") if isinstance(reward, dict) else None
    for row in rows:
        apply_reward_to_row(row, passed=passed, reward=reward_value)
    stem = f"bench-{task_name}"
    out_path = cfg.output.traces_dir / f"{stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if reward is not None:
        write_verification_sidecar(cfg.output.traces_dir, stem, reward)
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
        trial, result = asyncio.run(_create_and_run(config))
        exc_info = getattr(result, "exception_info", None)
        if console is not None and exc_info:
            exc_type = (
                exc_info.get("exception_type") if isinstance(exc_info, dict)
                else getattr(exc_info, "exception_type", None)
            ) or "agent error"
            console.print(f"[yellow]bench: {task_dir.name}: agent did not finish cleanly ({exc_type})[/yellow]")
        reward = _reward_from_result(result) or _reward_from_files(
            _agent_dir(trial).parent if _agent_dir(trial) else None
        )
        paths = _harvest_trace(cfg, trial, reward, task_dir.name)
        if not paths:
            if console is not None:
                console.print(f"[yellow]bench: no trace harvested for {task_dir.name}[/yellow]")
            continue
        if console is not None:
            label = f"reward={reward['reward']:g}" if reward else "no reward"
            console.print(f"[green]bench: {task_dir.name}: wrote {len(paths)} file(s) ({label})[/green]")
        written.extend(paths)
    return written
