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
    assert bench.repo is None
    assert bench.version is None
    assert bench.backend == "docker"


def test_bench_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "bench:\n  source: org/set\n  repo: https://github.com/o/r\n  version: '2.0'\n  backend: docker\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(config_file)
    assert cfg.bench.source == "org/set"
    assert cfg.bench.repo == "https://github.com/o/r"
    assert cfg.bench.version == "2.0"
    assert cfg.bench.backend == "docker"


def test_classify_remote_source():
    c = bench_runner._classify_remote_source
    assert c("terminal-bench@2.0", None, None) == ("registry", "terminal-bench@2.0")
    assert c("terminal-bench", None, "2.0") == ("registry", "terminal-bench@2.0")
    assert c("terminal-bench", None, None) == ("registry", "terminal-bench")
    assert c("org/name", None, None) == ("package", "org/name@latest")
    assert c("org/name@ref", None, None) == ("package", "org/name@ref")
    assert c("dataset", "https://github.com/o/r", None) == ("repo", "dataset")
    # version encoded in source wins over the version field
    assert c("terminal-bench@2.0", None, "9.9") == ("registry", "terminal-bench@2.0")


def test_bench_source_slug():
    assert bench_runner._bench_source_slug("terminal-bench@2.0", None) == "terminal-bench-2.0"
    assert bench_runner._bench_source_slug("org/name@ref", None) == "org-name-ref"
    assert bench_runner._bench_source_slug("terminal-bench", "2.0") == "terminal-bench-2.0"


def test_resolve_bench_source_local_passthrough(tmp_path, monkeypatch):
    # An existing local path is returned as-is and never triggers a download.
    local = tmp_path / "tasks"
    local.mkdir()
    monkeypatch.setattr(
        bench_runner, "_fetch_remote_source",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not download a local source")),
    )
    cfg = Config(bench={"source": str(local)}, output={"traces_dir": tmp_path / "out"})
    assert bench_runner._resolve_bench_source(cfg) == local


def test_resolve_bench_source_downloads_then_reuses(tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = Config(bench={"source": "terminal-bench@2.0"}, output={"traces_dir": out})
    calls = []

    def fake_fetch(cfg_arg, cache_dir):
        calls.append(cache_dir)
        # mimic harbor's export layout: <cache>/<dataset>/<task>/task.toml
        for task in ("task-a", "task-b"):
            d = cache_dir / "terminal-bench" / task
            d.mkdir(parents=True, exist_ok=True)
            (d / "task.toml").write_text("version='1.0'\n", encoding="utf-8")

    monkeypatch.setattr(bench_runner, "_fetch_remote_source", fake_fetch)

    root = bench_runner._resolve_bench_source(cfg)
    # bench intermediates default to a sibling `bench` dir next to traces_dir (out).
    assert root == out.parent / "bench" / "sources" / "terminal-bench-2.0" / "terminal-bench"
    assert sorted(d.name for d in bench_runner._resolve_task_dirs(root)) == ["task-a", "task-b"]
    assert len(calls) == 1

    # Second call reuses the cache (no re-download)...
    bench_runner._resolve_bench_source(cfg)
    assert len(calls) == 1
    # ...but --refresh forces a re-download.
    bench_runner._resolve_bench_source(cfg, refresh=True)
    assert len(calls) == 2


def test_resolve_bench_source_errors_when_download_yields_no_tasks(tmp_path, monkeypatch):
    cfg = Config(bench={"source": "bogus@1.0"}, output={"traces_dir": tmp_path / "out"})
    monkeypatch.setattr(bench_runner, "_fetch_remote_source", lambda *a, **k: None)  # writes nothing
    with pytest.raises(RuntimeError, match="no Harbor tasks found"):
        bench_runner._resolve_bench_source(cfg)


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


class _FakeResult:
    def __init__(self, verifier_result=None, exception_info=None):
        self.verifier_result = verifier_result
        self.exception_info = exception_info


class _FakeTrial:
    def __init__(self, agent_dir):
        self.paths = type("P", (), {"agent_dir": agent_dir})()


def test_rewards_from_result_preserves_full_dict():
    # The full multi-component rewards dict is kept (no clamping to a single scalar).
    r = _FakeResult(verifier_result={"rewards": {"reward": 1.0, "sub": 0.5}})
    assert bench_runner._rewards_from_result(r) == {"reward": 1.0, "sub": 0.5}
    assert bench_runner._rewards_from_result(_FakeResult(verifier_result=None)) is None
    assert bench_runner._rewards_from_result(_FakeResult(verifier_result={"rewards": {}})) is None
    # non-numeric / bool components are dropped; the numeric ones survive.
    assert bench_runner._rewards_from_result(
        _FakeResult(verifier_result={"rewards": {"reward": 0.0, "ok": True, "note": "x"}})
    ) == {"reward": 0.0}


def test_rewards_from_files_dict_and_text(tmp_path):
    wrapped = tmp_path / "w"
    wrapped.mkdir()
    (wrapped / "rewards.json").write_text(json.dumps({"rewards": {"reward": 1.0, "sub": 0.5}}), encoding="utf-8")
    assert bench_runner._rewards_from_files(wrapped) == {"reward": 1.0, "sub": 0.5}
    flat = tmp_path / "f"
    flat.mkdir()
    (flat / "reward.json").write_text(json.dumps({"score": 0.6}), encoding="utf-8")
    assert bench_runner._rewards_from_files(flat) == {"score": 0.6}
    txt = tmp_path / "t"
    txt.mkdir()
    (txt / "reward.txt").write_text("0.0\n", encoding="utf-8")
    assert bench_runner._rewards_from_files(txt) == {"reward": 0.0}
    bad = tmp_path / "b"
    bad.mkdir()
    (bad / "reward.txt").write_text("not-a-number\n", encoding="utf-8")
    assert bench_runner._rewards_from_files(bad) is None


def test_primary_score_and_route_split():
    assert bench_runner._primary_score({"reward": 1.0, "sub": 0.5}) == 1.0  # 'reward' key preferred
    assert bench_runner._primary_score({"score": 0.6}) == 0.6               # else first numeric
    assert bench_runner._primary_score(None) is None
    assert bench_runner._route_split(1.0) == "passed"
    assert bench_runner._route_split(0.0) == "failed"
    assert bench_runner._route_split(None) == "failed"      # unscored -> failed
    assert bench_runner._route_split(0.6) == "borderline"
    assert bench_runner._route_split(2.0) == "borderline"


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


def test_harvest_trace_pi_stream_writes_native_and_metadata(tmp_path):
    cfg = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "pi.txt").write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
    paths, split = bench_runner._harvest_trace(
        cfg, _FakeTrial(agent_dir), {"reward": 1.0, "sub": 0.5}, "add-bug"
    )
    assert split == "passed"
    assert paths == [tmp_path / "output" / "passed" / "bench-add-bug.jsonl"]
    # output is the PLAIN native trace (pi session events), NOT pre-converted training rows.
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    types = {row.get("type") for row in rows}
    assert "session" in types and "message" in types
    assert all("messages" not in row and "prompt" not in row for row in rows)
    # per-task metadata sidecar: full rewards dict (no clamping) + recovered trace metadata.
    meta = json.loads((tmp_path / "output" / "metadata" / "bench-add-bug.json").read_text(encoding="utf-8"))
    assert meta["split"] == "passed" and meta["reward"] == 1.0
    assert meta["rewards"] == {"reward": 1.0, "sub": 0.5}
    assert meta["agent"] == "pi" and meta["model"] == "z-ai/glm-5.2"
    # the normalized native session is kept under the sibling bench/ dir.
    assert (tmp_path / "bench" / "sessions" / "add-bug" / "bench-add-bug.jsonl").is_file()


def test_harvest_trace_routes_by_primary_score(tmp_path):
    cfg = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    for task, rewards, expected in [
        ("zero", {"reward": 0.0}, "failed"),
        ("unscored", None, "failed"),
        ("frac", {"reward": 0.6}, "borderline"),
    ]:
        agent_dir = tmp_path / task / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "pi.txt").write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
        paths, split = bench_runner._harvest_trace(cfg, _FakeTrial(agent_dir), rewards, task)
        assert split == expected
        assert paths == [tmp_path / "output" / expected / f"bench-{task}.jsonl"]


def test_harvest_trace_native_session_dir(tmp_path):
    cfg = Config(agent={"provider": "codex"}, output={"traces_dir": tmp_path / "output"})
    agent_dir = tmp_path / "trial" / "agent"
    _write_codex_session(agent_dir / "sessions")
    paths, split = bench_runner._harvest_trace(cfg, _FakeTrial(agent_dir), {"reward": 0.0}, "codex-task")
    assert split == "failed"
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    # native codex events preserved verbatim (not converted to training rows).
    assert any(row.get("type") in ("session_meta", "response_item") for row in rows)


def test_harvest_trace_no_trace_returns_empty(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    assert bench_runner._harvest_trace(cfg, _FakeTrial(agent_dir), None, "empty") == ([], None)


def test_bench_existing_output_finds_across_splits(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    assert bench_runner._bench_existing_output(cfg, "x") is None
    routed = tmp_path / "output" / "borderline" / "bench-x.jsonl"
    routed.parent.mkdir(parents=True)
    routed.write_text("{}\n", encoding="utf-8")
    assert bench_runner._bench_existing_output(cfg, "x") == routed


def test_run_bench_resume_skips_already_harvested(tmp_path):
    pytest.importorskip("harbor")
    # A task already harvested into a split -> resume skips it without invoking harbor.
    tasks_dir = tmp_path / "tasks"
    task = tasks_dir / "add-bug"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("", encoding="utf-8")
    out = tmp_path / "output"
    existing = out / "passed" / "bench-add-bug.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"type": "session"}\n', encoding="utf-8")

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
