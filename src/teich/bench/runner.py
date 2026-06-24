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
import os
import re
import shutil
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

def _bench_root(cfg: Config) -> Path:
    """Working dir for harbor's raw trials, downloaded sources, and normalized sessions.

    Defaults to a ``bench`` directory beside ``traces_dir`` (parallel to sandbox/failures,
    never inside the dataset); overridable via ``output.bench_dir``.
    """
    if cfg.output.bench_dir is not None:
        return Path(cfg.output.bench_dir)
    return cfg.output.traces_dir.parent / "bench"

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

    For an OpenRouter project the same key is exported under both names because the
    in-container agent picks the var by *agent* type, not provider: pi/hermes read
    ``OPENROUTER_API_KEY`` while codex/claude-code use ``OPENAI_API_KEY`` against the
    OpenRouter ``base_url``. For a plain ``openai`` project only ``OPENAI_API_KEY`` is
    set, so an OpenAI key is never leaked under the OpenRouter name.
    """
    env: dict[str, str] = {}
    api_key = cfg.get_api_key()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
        if cfg.api.provider == "openrouter":
            env["OPENROUTER_API_KEY"] = api_key
    base_url = cfg.get_base_url()
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    return env


def _resolve_task_dirs(source: str | Path) -> list[Path]:
    """Resolve a local Harbor task root to one or more task directories.

    Accepts a single-task dir (has ``task.toml``) or a dir of task dirs. Remote
    specs are turned into a local dir by ``_resolve_bench_source`` before this runs.
    """
    path = Path(source).expanduser()
    if not path.exists():
        raise RuntimeError(f"bench task directory not found: {path}.")
    if (path / "task.toml").is_file():
        return [path]
    tasks = sorted(d for d in path.iterdir() if d.is_dir() and (d / "task.toml").is_file())
    if not tasks:
        raise RuntimeError(f"No Harbor tasks (a task.toml) found under {path}.")
    return tasks


def _classify_remote_source(source: str, repo: str | None, version: str | None) -> tuple[str, str]:
    """Map a remote ``bench.source`` to (client kind, harbor dataset ref).

    Mirrors harbor's ``download`` resolution: an explicit ``repo`` (git/HF registry)
    wins, else ``org/name`` is a package-registry dataset, else a legacy-registry name.
    A version already encoded in ``source`` as ``name@version`` takes precedence over
    the ``bench.version`` field.
    """
    has_version = "@" in source
    name = source.split("@", 1)[0]

    def _ref(default_version: str | None = None) -> str:
        if has_version:
            return source
        if version:
            return f"{source}@{version}"
        return f"{source}@{default_version}" if default_version else source

    if repo:
        return "repo", _ref()
    if "/" in name:
        return "package", _ref(default_version="latest")
    return "registry", _ref()


def _bench_source_slug(source: str, version: str | None) -> str:
    """Filesystem-safe cache slug for a remote spec (e.g. ``terminal-bench@2.0`` -> ``terminal-bench-2.0``)."""
    raw = source if "@" in source else (f"{source}@{version}" if version else source)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-") or "source"


def _task_root(cache_dir: Path) -> Path | None:
    """The local dir to hand to ``_resolve_task_dirs`` for a downloaded source.

    harbor exports tasks as ``<cache>/<dataset>/<task>/task.toml``; return the common
    parent of every ``task.toml`` (the dataset dir, or the single task dir). None if empty.
    """
    parents = sorted({toml.parent for toml in cache_dir.rglob("task.toml")})
    if not parents:
        return None
    if len(parents) == 1:
        return parents[0]
    return Path(os.path.commonpath([str(p) for p in parents]))


async def _download_remote_source_async(cfg: Config, cache_dir: Path) -> None:
    """Download a remote ``bench.source`` into ``cache_dir`` via harbor's registry client."""
    kind, ref = _classify_remote_source(
        (cfg.bench.source or "").strip(), cfg.bench.repo, cfg.bench.version
    )
    if kind == "repo":
        from harbor.registry.client.factory import RegistryClientFactory

        client = RegistryClientFactory.create(repo=cfg.bench.repo)
    elif kind == "package":
        from harbor.registry.client.package import PackageDatasetClient

        client = PackageDatasetClient()
    else:
        from harbor.registry.client.factory import RegistryClientFactory

        client = RegistryClientFactory.create()
    cache_dir.mkdir(parents=True, exist_ok=True)
    await client.download_dataset(ref, overwrite=True, output_dir=cache_dir, export=True)


def _fetch_remote_source(cfg: Config, cache_dir: Path) -> None:
    """Sync wrapper around the async harbor download (its own asyncio.run)."""
    try:
        asyncio.run(_download_remote_source_async(cfg, cache_dir))
    except Exception as exc:  # network / unknown dataset / auth -> a clean bench error
        raise RuntimeError(
            f"Failed to download bench.source {cfg.bench.source!r}: {type(exc).__name__}: {exc}. "
            "Check the spec/version, network, and (for HF/private registries) HF_TOKEN."
        ) from exc


def _resolve_bench_source(cfg: Config, *, refresh: bool = False) -> Path:
    """Return a local task root for ``bench.source``: a local path as-is, else download it."""
    source = (cfg.bench.source or "").strip()
    local = Path(source).expanduser()
    if local.exists():
        return local
    cache_dir = _bench_root(cfg) / "sources" / _bench_source_slug(source, cfg.bench.version)
    root = None if refresh else _task_root(cache_dir)
    if root is None:
        if refresh and cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        _fetch_remote_source(cfg, cache_dir)
        root = _task_root(cache_dir)
    if root is None:
        raise RuntimeError(
            f"bench.source {source!r}: no Harbor tasks found after downloading into {cache_dir}."
        )
    return root


def _bench_model_name(cfg: Config) -> str | None:
    """Model name for the in-container agent, with harbor's provider prefix when needed.

    harbor's pi agent requires ``<provider>/<model>`` and splits on the first ``/`` to
    pick the credential env var. A config like ``model: z-ai/glm-5.2`` with
    ``api.provider: openrouter`` would otherwise be read as provider ``z-ai`` (no key,
    no call), so we prefix the api provider when it isn't already there.
    """
    model = cfg.get_effective_model()
    if not model:
        return model
    api_provider = (cfg.api.provider or "").strip()
    if cfg.get_agent_provider() == "pi" and api_provider and not model.startswith(f"{api_provider}/"):
        return f"{api_provider}/{model}"
    return model


def _build_trial_config(cfg: Config, task_dir: Path, trials_dir: Path) -> Any:
    """Build a harbor TrialConfig from teich config (agent provider, model, api auth, backend)."""
    from harbor.models.agent.name import AgentName
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import TaskConfig, TrialConfig

    config = TrialConfig(task=TaskConfig(path=task_dir), trials_dir=trials_dir)
    config.agent.name = AgentName(_agent_name_for(cfg.get_agent_provider()))
    model = _bench_model_name(cfg)
    if model:
        config.agent.model_name = model
    config.agent.env.update(_agent_auth_env(cfg))
    try:
        config.environment.type = EnvironmentType(cfg.bench.backend)
    except ValueError as exc:
        supported = ", ".join(t.value for t in EnvironmentType)
        raise RuntimeError(
            f"Unknown bench.backend {cfg.bench.backend!r}; harbor supports: {supported}."
        ) from exc
    return config


def _reward_dict_from_value(value: Any) -> dict[str, Any] | None:
    """Wrap a single numeric reward into our ``{reward, passed}`` shape.

    ``passed`` is ``reward > 0``: any positive score (including partial credit)
    counts as passed, which is looser than prompts mode, where ``passed`` is the
    verifier exit code / a full fail-to-pass-pass-to-pass transition.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return {"reward": float(value), "passed": float(value) > 0}


def _reward_from_mapping(data: Any) -> dict[str, Any] | None:
    """Normalize any of harbor's reward-dict shapes into our ``{reward, passed}``.

    Accepts ``{"reward": <num>}``, the verifier's ``{"rewards": {...}}`` wrapper, or
    a flat ``{name: <num>}`` map; prefers a ``reward`` key, else the first numeric
    value. Returns None when no numeric reward is present. Normalizing here keeps
    every reward source on the same contract, so callers never see a raw harbor
    dict that lacks a ``reward``/``passed``.
    """
    if not isinstance(data, dict):
        return None
    rewards = data.get("rewards") if isinstance(data.get("rewards"), dict) else data
    reward = _reward_dict_from_value(rewards.get("reward"))
    if reward is not None:
        return reward
    for value in rewards.values():
        reward = _reward_dict_from_value(value)
        if reward is not None:
            return reward
    return None


def _reward_from_sidecar(reward_path: Path) -> dict[str, Any] | None:
    """Parse harbor's reward.json/rewards.json sidecar into our ``{reward, passed}``."""
    try:
        data = json.loads(reward_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _reward_from_mapping(data)


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
    return _reward_dict_from_value(value)


def _reward_from_result(result: Any) -> dict[str, Any] | None:
    """Read the reward from harbor's ``TrialResult.verifier_result`` (the canonical source).

    harbor reports ``verifier_result = {"rewards": {"reward": <float>, ...}}``; we
    take ``reward`` when present, else the first numeric reward value.
    """
    verifier = getattr(result, "verifier_result", None)
    if isinstance(verifier, dict):
        return _reward_from_mapping(verifier)
    return _reward_from_mapping(getattr(verifier, "rewards", None))


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
            norm_dir = _bench_root(cfg) / "sessions" / task_name
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


def _bench_stem(task_name: str) -> str:
    """Dataset filename stem for a bench task (``bench-<task>`` -> ``bench-<task>.jsonl``)."""
    return f"bench-{task_name}"


def _bench_output_path(cfg: Config, task_name: str) -> Path:
    return cfg.output.traces_dir / f"{_bench_stem(task_name)}.jsonl"


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
    out_path = _bench_output_path(cfg, task_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if reward is not None:
        write_verification_sidecar(cfg.output.traces_dir, _bench_stem(task_name), reward)
    return [out_path]


async def _create_and_run(config: Any) -> tuple[Any, Any]:
    from harbor.trial.trial import Trial

    trial = await Trial.create(config)
    result = await trial.run()
    return trial, result


def run_bench(
    cfg: Config, *, console: Any = None, resume: bool = False, refresh: bool = False
) -> list[Path]:
    """Run benchmark tasks from ``cfg.bench.source`` and write reward-labeled traces."""
    source = (cfg.bench.source or "").strip()
    if not source:
        raise RuntimeError(
            "--mode bench requires bench.source in config (a local Harbor task directory or "
            "dir of task dirs, a registry spec like 'terminal-bench@2.0' or 'org/name@ref', "
            "or a git/HF registry via bench.repo)."
        )
    _require_harbor()
    source_dir = _resolve_bench_source(cfg, refresh=refresh)
    if console is not None and source_dir != Path(source).expanduser():
        console.print(f"[blue]bench: resolved {source} -> {source_dir}[/blue]")
    task_dirs = _resolve_task_dirs(source_dir)
    trials_dir = _bench_root(cfg) / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task_dir in task_dirs:
        existing = _bench_output_path(cfg, task_dir.name)
        if resume and existing.is_file() and existing.stat().st_size > 0:
            if console is not None:
                console.print(f"[yellow]bench: skipping {task_dir.name} (already harvested)[/yellow]")
            written.append(existing)
            continue
        if console is not None:
            console.print(f"[blue]bench: running {task_dir.name}[/blue]")
        # Config errors (bad provider/backend) are the same for every task, so let them
        # abort the run (the CLI turns RuntimeError into a clean message).
        config = _build_trial_config(cfg, task_dir, trials_dir)
        try:
            trial, result = asyncio.run(_create_and_run(config))
        except Exception as exc:  # this task's failure (Docker build, harbor, etc.) — skip it
            if console is not None:
                console.print(f"[red]bench: {task_dir.name}: failed ({type(exc).__name__}: {exc})[/red]")
            continue
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
            label = f"reward={reward.get('reward'):g}" if reward else "no reward"
            console.print(f"[green]bench: {task_dir.name}: wrote {len(paths)} file(s) ({label})[/green]")
        written.extend(paths)
    return written
