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


def test_agent_auth_env_openai_provider_does_not_set_openrouter_key():
    # An OpenAI key must not be smeared into OPENROUTER_API_KEY.
    cfg = Config(api={"provider": "openai", "api_key": "sk-openai"})
    env = bench_runner._agent_auth_env(cfg)
    assert env["OPENAI_API_KEY"] == "sk-openai"
    assert "OPENROUTER_API_KEY" not in env


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


def test_ingest_session_dir_empty_returns_nothing(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    empty = tmp_path / "logs" / "sessions"
    empty.mkdir(parents=True)
    assert bench_runner._ingest_session_dir(cfg, empty, {"reward": 1.0, "passed": True}, "x") == []
    assert not (tmp_path / "output" / "bench-x.jsonl").exists()


class _FakeResult:
    def __init__(self, verifier_result=None, exception_info=None):
        self.verifier_result = verifier_result
        self.exception_info = exception_info


def test_reward_from_result_reads_verifier_rewards():
    result = _FakeResult(verifier_result={"rewards": {"reward": 1.0}})
    assert bench_runner._reward_from_result(result) == {"reward": 1.0, "passed": True}
    zero = _FakeResult(verifier_result={"rewards": {"reward": 0.0}})
    assert bench_runner._reward_from_result(zero) == {"reward": 0.0, "passed": False}


def test_reward_from_result_falls_back_and_handles_missing():
    # No "reward" key -> first numeric value wins.
    assert bench_runner._reward_from_result(
        _FakeResult(verifier_result={"rewards": {"score": 0.5}})
    ) == {"reward": 0.5, "passed": True}
    # No usable reward -> None (so the caller can fall back to on-disk files).
    assert bench_runner._reward_from_result(_FakeResult(verifier_result=None)) is None
    assert bench_runner._reward_from_result(_FakeResult(verifier_result={"rewards": {}})) is None


def test_reward_from_result_multi_key_precedence():
    # A "reward" key wins over other numeric components...
    assert bench_runner._reward_from_result(
        _FakeResult(verifier_result={"rewards": {"penalty": 1.0, "reward": 0.0}})
    ) == {"reward": 0.0, "passed": False}
    # ...otherwise the first numeric value (insertion order) is taken.
    assert bench_runner._reward_from_result(
        _FakeResult(verifier_result={"rewards": {"a": 0.0, "b": 0.5}})
    ) == {"reward": 0.0, "passed": False}


def test_reward_from_files_normalizes_each_sidecar_shape(tmp_path):
    # reward.txt single value
    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "reward.txt").write_text("1.0\n", encoding="utf-8")
    assert bench_runner._reward_from_files(txt) == {"reward": 1.0, "passed": True}
    # rewards.json with the {"rewards": {...}} wrapper
    wrapped = tmp_path / "wrapped"
    wrapped.mkdir()
    (wrapped / "rewards.json").write_text(json.dumps({"rewards": {"reward": 0.0}}), encoding="utf-8")
    assert bench_runner._reward_from_files(wrapped) == {"reward": 0.0, "passed": False}
    # reward.json as a flat multi-key map with no "reward" key -> first numeric
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "reward.json").write_text(json.dumps({"score": 0.5}), encoding="utf-8")
    assert bench_runner._reward_from_files(flat) == {"reward": 0.5, "passed": True}


def test_reward_parsers_reject_non_numeric(tmp_path):
    (tmp_path / "reward.txt").write_text("not-a-number\n", encoding="utf-8")
    assert bench_runner._reward_from_text(tmp_path / "reward.txt") is None
    (tmp_path / "reward.json").write_text(json.dumps({"reward": True}), encoding="utf-8")  # bool != numeric
    assert bench_runner._reward_from_sidecar(tmp_path / "reward.json") is None
    (tmp_path / "empty.json").write_text(json.dumps({}), encoding="utf-8")
    assert bench_runner._reward_from_sidecar(tmp_path / "empty.json") is None


# A minimal pi `--mode json` stream (as harbor's --no-session pi run emits to pi.txt).
_PI_STREAM_LINES = [
    'Warning: Model "z-ai/glm-5.2" not found for provider "openrouter". Using custom model id.',
    json.dumps({"type": "session", "version": 3, "id": "abc", "cwd": "/app"}),
    json.dumps({"type": "agent_start"}),
    json.dumps({"type": "message_end", "message": {"role": "user",
        "content": [{"type": "text", "text": "Fix add()"}]}}),
    json.dumps({"type": "message_end", "message": {"role": "assistant", "provider": "openrouter",
        "model": "z-ai/glm-5.2", "content": [
            {"type": "thinking", "thinking": "read it"},
            {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "/app/app.py"}}]}}),
    json.dumps({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "read",
        "result": {"content": [{"type": "text", "text": "return a - b"}]}, "isError": False}),
    json.dumps({"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c1",
        "toolName": "read", "content": [{"type": "text", "text": "return a - b"}], "isError": False}}),
    json.dumps({"type": "message_end", "message": {"role": "assistant",
        "content": [{"type": "text", "text": "Fixed."}]}}),
]


def test_pi_stream_to_session_events(tmp_path):
    pi_txt = tmp_path / "pi.txt"
    pi_txt.write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
    events = bench_runner._pi_stream_to_session_events(pi_txt)
    types = [e["type"] for e in events]
    # Warning line skipped; session kept; message_end -> message; a model_change is
    # synthesized just before the first assistant message (converter captures it anywhere).
    assert types == ["session", "message", "model_change", "message", "message", "message"]
    model_change = next(e for e in events if e["type"] == "model_change")
    assert model_change == {"type": "model_change", "provider": "openrouter", "modelId": "z-ai/glm-5.2"}
    assert all(e["message"]["role"] for e in events if e["type"] == "message")


def test_pi_stream_without_provider_model_omits_model_change(tmp_path):
    # When the first assistant message carries no provider/model, no model_change is
    # synthesized, but the messages still convert.
    lines = [
        json.dumps({"type": "session", "id": "x", "cwd": "/app"}),
        json.dumps({"type": "message_end", "message": {"role": "user",
            "content": [{"type": "text", "text": "hi"}]}}),
        json.dumps({"type": "message_end", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "done"}]}}),
    ]
    pi_txt = tmp_path / "pi.txt"
    pi_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    events = bench_runner._pi_stream_to_session_events(pi_txt)
    assert [e["type"] for e in events] == ["session", "message", "message"]
    assert not any(e["type"] == "model_change" for e in events)


class _FakeTrial:
    def __init__(self, agent_dir):
        self.paths = type("P", (), {"agent_dir": agent_dir})()


def test_harvest_trace_pi_stream_produces_rewarded_rows(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "pi.txt").write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
    written = bench_runner._harvest_trace(
        cfg, _FakeTrial(agent_dir), {"reward": 1.0, "passed": True}, "add-bug"
    )
    assert written == [tmp_path / "output" / "bench-add-bug.jsonl"]
    rows = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["reward"] == 1.0 and rows[0]["passed"] is True
    roles = [m["role"] for m in rows[0]["messages"]]
    assert "user" in roles and "assistant" in roles and "tool" in roles
    # The normalized session file is kept under the hidden output/.bench dir (excluded
    # from the dataset card / publish) for inspection.
    assert (tmp_path / "output" / ".bench" / "sessions" / "add-bug" / "pi.jsonl").is_file()


def test_harvest_trace_prefers_native_session_dir(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    agent_dir = tmp_path / "trial" / "agent"
    _write_codex_session(agent_dir / "sessions")
    written = bench_runner._harvest_trace(
        cfg, _FakeTrial(agent_dir), {"reward": 0.0, "passed": False}, "codex-task"
    )
    rows = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["reward"] == 0.0 and rows[0]["passed"] is False


def test_harvest_trace_no_trace_returns_empty(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    assert bench_runner._harvest_trace(cfg, _FakeTrial(agent_dir), None, "empty") == []


def test_bench_output_path_uses_stem(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    assert bench_runner._bench_output_path(cfg, "add-bug") == tmp_path / "output" / "bench-add-bug.jsonl"


def test_run_bench_resume_skips_already_harvested(tmp_path):
    pytest.importorskip("harbor")
    # A task whose output already exists -> resume skips it without invoking harbor.
    tasks_dir = tmp_path / "tasks"
    task = tasks_dir / "add-bug"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("", encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir()
    existing = out / "bench-add-bug.jsonl"
    existing.write_text('{"messages": [], "reward": 1.0, "passed": true}\n', encoding="utf-8")

    cfg = Config(
        agent={"provider": "pi"},
        model={"model": "openrouter/z-ai/glm-5.2"},
        api={"provider": "openrouter", "api_key": "sk-test"},
        bench={"source": str(tasks_dir)},
        output={"traces_dir": out},
    )
    # No harbor Trial is created for a skipped task; if it were, this would fail (sk-test).
    written = bench_runner.run_bench(cfg, resume=True)
    assert written == [existing]


class _RecordingConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, message=""):
        self.lines.append(str(message))


def test_run_bench_reports_agent_failure_and_empty_harvest(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    tasks_dir = tmp_path / "tasks"
    task = tasks_dir / "add-bug"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("", encoding="utf-8")
    # A trial whose agent dir is empty -> nothing to harvest.
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)

    async def _fake_create_and_run(config):
        result = _FakeResult(verifier_result=None, exception_info={"exception_type": "NonZeroAgentExitCodeError"})
        return _FakeTrial(agent_dir), result

    monkeypatch.setattr(bench_runner, "_create_and_run", _fake_create_and_run)
    cfg = Config(
        agent={"provider": "pi"},
        model={"model": "openrouter/z-ai/glm-5.2"},
        api={"provider": "openrouter", "api_key": "sk-test"},
        bench={"source": str(tasks_dir)},
        output={"traces_dir": tmp_path / "output"},
    )
    console = _RecordingConsole()
    written = bench_runner.run_bench(cfg, console=console, resume=False)
    assert written == []
    blob = "\n".join(console.lines)
    assert "did not finish cleanly (NonZeroAgentExitCodeError)" in blob
    assert "no trace harvested for add-bug" in blob


def test_run_bench_continues_past_one_task_infra_failure(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    tasks_dir = tmp_path / "tasks"
    for name in ("a-task", "b-task"):
        (tasks_dir / name).mkdir(parents=True)
        (tasks_dir / name / "task.toml").write_text("", encoding="utf-8")

    calls: list[str] = []

    async def _fake_create_and_run(config):
        calls.append(str(config.task.path))
        # First task blows up with a non-RuntimeError infra error; second must still run.
        if len(calls) == 1:
            raise OSError("docker build failed")
        return _FakeTrial(tmp_path / "empty"), _FakeResult(verifier_result={"rewards": {"reward": 1.0}})

    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(bench_runner, "_create_and_run", _fake_create_and_run)
    cfg = Config(
        agent={"provider": "pi"},
        model={"model": "openrouter/z-ai/glm-5.2"},
        api={"provider": "openrouter", "api_key": "sk-test"},
        bench={"source": str(tasks_dir)},
        output={"traces_dir": tmp_path / "output"},
    )
    console = _RecordingConsole()
    bench_runner.run_bench(cfg, console=console, resume=False)
    # Both tasks were attempted (the first failure did not abort the batch).
    assert len(calls) == 2
    assert any("failed (OSError" in line for line in console.lines)


def test_run_bench_skips_task_level_runtimeerror(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "t").mkdir(parents=True)
    (tasks_dir / "t" / "task.toml").write_text("", encoding="utf-8")

    async def _boom(config):
        raise RuntimeError("harbor blew up on this task")

    monkeypatch.setattr(bench_runner, "_create_and_run", _boom)
    cfg = Config(
        agent={"provider": "pi"},
        model={"model": "openrouter/z-ai/glm-5.2"},
        api={"provider": "openrouter", "api_key": "sk-test"},
        bench={"source": str(tasks_dir)},
        output={"traces_dir": tmp_path / "output"},
    )
    console = _RecordingConsole()
    # A RuntimeError from trial execution is a task failure, not a config error -> skip it.
    written = bench_runner.run_bench(cfg, console=console, resume=False)
    assert written == []
    assert any("failed (RuntimeError" in line for line in console.lines)


def test_run_bench_aborts_on_invalid_backend(tmp_path):
    pytest.importorskip("harbor")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "t").mkdir(parents=True)
    (tasks_dir / "t" / "task.toml").write_text("", encoding="utf-8")
    cfg = Config(
        agent={"provider": "pi"},
        model={"model": "openrouter/z-ai/glm-5.2"},
        api={"provider": "openrouter", "api_key": "sk-test"},
        bench={"source": str(tasks_dir), "backend": "dokcer"},  # typo -> config-level error
        output={"traces_dir": tmp_path / "output"},
    )
    # A bad backend applies to every task, so it aborts (RuntimeError) instead of skipping.
    with pytest.raises(RuntimeError, match="Unknown bench.backend"):
        bench_runner.run_bench(cfg, resume=False)
