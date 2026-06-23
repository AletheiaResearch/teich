"""Tests for bench-mode config, driver helpers, and the native-session ingest.

The live harbor `Trial` run is Docker/integration; here we cover the teich-side:
config, provider/auth mapping, task resolution, TrialConfig construction (when the
optional harbor extra is present), and the session->rows+reward ingest.
"""

import json

import pytest

from teich.bench import run_bench
from teich.bench import runner as bench_runner
from teich.config import Config


def test_bench_config_defaults():
    bench = Config().bench
    assert bench.source is None
    assert bench.backend == "docker"


def test_bench_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("bench:\n  source: ./tasks\n  backend: docker\n", encoding="utf-8")
    cfg = Config.from_yaml(config_file)
    assert cfg.bench.source == "./tasks"
    assert cfg.bench.backend == "docker"


def test_run_bench_requires_source():
    with pytest.raises(RuntimeError, match="requires bench.source"):
        run_bench(Config())


def test_agent_name_mapping():
    assert bench_runner._agent_name_for("codex") == "codex"
    assert bench_runner._agent_name_for("claude-code") == "claude-code"
    assert bench_runner._agent_name_for("claude") == "claude-code"
    assert bench_runner._agent_name_for("pi") == "pi"
    assert bench_runner._agent_name_for("hermes") == "hermes"
    with pytest.raises(RuntimeError, match="does not support agent provider"):
        bench_runner._agent_name_for("chat")


def test_agent_auth_env_from_api_config():
    cfg = Config(api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk-test"})
    env = bench_runner._agent_auth_env(cfg)
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["OPENROUTER_API_KEY"] == "sk-test"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_resolve_task_dirs_single_and_collection(tmp_path):
    single = tmp_path / "one"
    single.mkdir()
    (single / "task.toml").write_text("", encoding="utf-8")
    assert bench_runner._resolve_task_dirs(str(single)) == [single]

    coll = tmp_path / "many"
    (coll / "a").mkdir(parents=True)
    (coll / "a" / "task.toml").write_text("", encoding="utf-8")
    (coll / "b").mkdir(parents=True)
    (coll / "b" / "task.toml").write_text("", encoding="utf-8")
    assert bench_runner._resolve_task_dirs(str(coll)) == [coll / "a", coll / "b"]

    with pytest.raises(RuntimeError, match="not found"):
        bench_runner._resolve_task_dirs(str(tmp_path / "missing"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No Harbor tasks"):
        bench_runner._resolve_task_dirs(str(empty))


def test_build_trial_config_maps_provider_model_auth_backend(tmp_path):
    pytest.importorskip("harbor")
    cfg = Config(
        agent={"provider": "codex"},
        model={"model": "z-ai/glm-5.2"},
        api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk-test"},
    )
    task_dir = tmp_path / "t"
    task_dir.mkdir()
    config = bench_runner._build_trial_config(cfg, task_dir, tmp_path / "trials")
    assert config.agent.name.value == "codex"
    assert config.agent.model_name == "z-ai/glm-5.2"
    assert config.agent.env["OPENAI_API_KEY"] == "sk-test"
    assert config.agent.env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert config.environment.type.value == "docker"


def _write_codex_session(sessions_dir):
    sessions_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "session_meta", "payload": {"id": "s1"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Fix the bug"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Fixed it."}]}},
    ]
    (sessions_dir / "rollout.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


def test_ingest_session_dir_attaches_reward_and_writes(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    sessions = tmp_path / "logs" / "sessions"
    _write_codex_session(sessions)
    reward = {"passed": True, "exit_code": 0, "fail_to_pass": {"t::a": "passed"}}
    written = bench_runner._ingest_session_dir(cfg, sessions, reward, "widgets-bug-01")
    assert written == [tmp_path / "output" / "bench-widgets-bug-01.jsonl"]
    rows = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["passed"] is True and rows[0]["reward"] == 1.0
    sidecar = tmp_path / "output" / "verification" / "bench-widgets-bug-01.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["passed"] is True


def test_ingest_session_dir_without_reward(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    sessions = tmp_path / "logs" / "sessions"
    _write_codex_session(sessions)
    written = bench_runner._ingest_session_dir(cfg, sessions, None, "no-reward")
    rows = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and "reward" not in rows[0] and "passed" not in rows[0]
    assert not (tmp_path / "output" / "verification" / "bench-no-reward.json").exists()
