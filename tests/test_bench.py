"""Tests for bench mode: config (sources array), the shared backend base/harvest, the
harbor backend, and the driver loop. Live harbor/Docker runs stay integration-only; here we
cover config, reward/route/harvest, harbor resolution/normalization, and driver behavior."""

import json

import pytest

from teich.bench import run_bench
from teich.bench import runner as bench_runner
from teich.bench.backends import base, get_backend
from teich.bench.backends import harbor as hb
from teich.config import BenchSource, Config


# --------------------------------------------------------------------------- config

def test_bench_config_defaults():
    assert Config().bench.sources == []


def test_bench_sources_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "bench:\n"
        "  sources:\n"
        "    - { type: harbor, source: terminal-bench@2.0 }\n"
        "    - { type: swe-bench, source: SWE-bench/SWE-bench_Verified, split: test }\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(config_file)
    assert [s.type for s in cfg.bench.sources] == ["harbor", "swe-bench"]
    assert cfg.bench.sources[0].source == "terminal-bench@2.0"
    assert cfg.bench.sources[0].backend == "docker"          # default
    assert cfg.bench.sources[1].split == "test"


def test_run_bench_requires_sources():
    with pytest.raises(RuntimeError, match="bench.sources"):
        run_bench(Config())


def test_get_backend():
    assert get_backend("harbor").type == "harbor"
    with pytest.raises(RuntimeError, match="Unknown bench source type"):
        get_backend("nope")


# ------------------------------------------------------------------- base: scoring/routing

def test_numeric_primary_route():
    assert base.numeric(1) == 1.0 and base.numeric(True) is None and base.numeric("x") is None
    assert base.primary_score({"reward": 1.0, "sub": 0.5}) == 1.0   # 'reward' preferred
    assert base.primary_score({"score": 0.6}) == 0.6                # else first numeric
    assert base.primary_score(None) is None
    assert base.route_split(1.0) == "passed"
    assert base.route_split(0.0) == "failed"
    assert base.route_split(None) == "failed"
    assert base.route_split(0.6) == "borderline"


def test_rewards_from_mapping_keeps_full_dict():
    assert base.rewards_from_mapping({"rewards": {"reward": 1.0, "sub": 0.5, "bad": "x"}}) == {
        "reward": 1.0,
        "sub": 0.5,
    }
    assert base.rewards_from_mapping({"rewards": {}}) is None
    assert base.rewards_from_mapping(None) is None


def test_bench_stem_namespaced_by_source():
    s = BenchSource(type="harbor", source="terminal-bench@2.0")
    assert base.bench_stem(s, "task-a") == "bench-terminal-bench-2.0-task-a"
    s2 = BenchSource(type="swe-bench", source="SWE-bench/SWE-bench_Verified")
    assert base.bench_stem(s2, "astropy__astropy-12907") == (
        "bench-SWE-bench-SWE-bench_Verified-astropy__astropy-12907"
    )


def test_existing_output_across_splits(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    assert base.existing_output(cfg, "bench-x") is None
    routed = tmp_path / "output" / "borderline" / "bench-x.jsonl"
    routed.parent.mkdir(parents=True)
    routed.write_text("{}\n", encoding="utf-8")
    assert base.existing_output(cfg, "bench-x") == routed


def test_bench_root_sibling_of_output(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "out"})
    assert base.bench_root(cfg) == tmp_path / "bench"
    cfg2 = Config(output={"traces_dir": tmp_path / "out", "bench_dir": tmp_path / "custom"})
    assert base.bench_root(cfg2) == tmp_path / "custom"


# ------------------------------------------------------------------- base: harvest

def test_harvest_writes_native_trace_and_metadata(tmp_path):
    cfg = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source="terminal-bench@2.0")
    task = base.BenchTask(id="add-bug")
    run = base.BenchRun(
        native_lines=['{"type":"session","id":"s"}', '{"type":"message","message":{"role":"user","content":[]}}'],
        rewards={"reward": 0.6, "tests": 0.8},
        metadata={"model": "z-ai/glm-5.2", "exception": None},
    )
    paths, split = base.harvest(cfg, source, task, run)
    stem = base.bench_stem(source, "add-bug")
    assert split == "borderline"
    assert paths == [tmp_path / "output" / "borderline" / f"{stem}.jsonl"]
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row.get("type") for row in rows} == {"session", "message"}  # native, not converted
    meta = json.loads((tmp_path / "output" / "metadata" / f"{stem}.json").read_text(encoding="utf-8"))
    assert meta["split"] == "borderline" and meta["reward"] == 0.6
    assert meta["rewards"] == {"reward": 0.6, "tests": 0.8}            # full dict, no clamping
    assert meta["source"] == "terminal-bench-2.0" and meta["type"] == "harbor"
    assert meta["agent"] == "pi" and meta["model"] == "z-ai/glm-5.2"


# ------------------------------------------------------------------- harbor backend helpers

def test_harbor_agent_name_mapping():
    assert hb._agent_name_for("codex") == "codex"
    assert hb._agent_name_for("claude-code") == "claude-code"
    assert hb._agent_name_for("claude") == "claude-code"
    assert hb._agent_name_for("pi") == "pi"
    assert hb._agent_name_for("hermes") == "hermes"
    with pytest.raises(RuntimeError, match="does not support agent provider"):
        hb._agent_name_for("chat")


def test_harbor_auth_env():
    env = hb._agent_auth_env(
        Config(api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk"})
    )
    assert env["OPENAI_API_KEY"] == "sk" and env["OPENROUTER_API_KEY"] == "sk"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    env2 = hb._agent_auth_env(Config(api={"provider": "openai", "api_key": "sk-o"}))
    assert env2["OPENAI_API_KEY"] == "sk-o" and "OPENROUTER_API_KEY" not in env2


def test_harbor_model_prefix():
    pi_or = Config(agent={"provider": "pi"}, model={"model": "z-ai/glm-5.2"}, api={"provider": "openrouter"})
    assert hb._bench_model_name(pi_or) == "openrouter/z-ai/glm-5.2"
    already = Config(agent={"provider": "pi"}, model={"model": "openrouter/z-ai/glm-5.2"}, api={"provider": "openrouter"})
    assert hb._bench_model_name(already) == "openrouter/z-ai/glm-5.2"
    codex = Config(agent={"provider": "codex"}, model={"model": "z-ai/glm-5.2"}, api={"provider": "openrouter"})
    assert hb._bench_model_name(codex) == "z-ai/glm-5.2"


def test_harbor_classify_remote_source():
    assert hb._classify_remote_source("terminal-bench@2.0", None, None) == ("registry", "terminal-bench@2.0")
    assert hb._classify_remote_source("terminal-bench", None, "2.0") == ("registry", "terminal-bench@2.0")
    assert hb._classify_remote_source("org/name", None, None) == ("package", "org/name@latest")
    assert hb._classify_remote_source("org/name@ref", None, None) == ("package", "org/name@ref")
    assert hb._classify_remote_source("ds", "https://github.com/o/r", None) == ("repo", "ds")


def test_harbor_resolve_task_dirs(tmp_path):
    single = tmp_path / "one"
    single.mkdir()
    (single / "task.toml").write_text("", encoding="utf-8")
    assert hb._resolve_task_dirs(single) == [single]
    coll = tmp_path / "many"
    (coll / "a").mkdir(parents=True)
    (coll / "a" / "task.toml").write_text("", encoding="utf-8")
    (coll / "b").mkdir()
    (coll / "b" / "task.toml").write_text("", encoding="utf-8")
    assert hb._resolve_task_dirs(coll) == [coll / "a", coll / "b"]
    with pytest.raises(RuntimeError, match="No Harbor tasks"):
        empty = tmp_path / "empty"
        empty.mkdir()
        hb._resolve_task_dirs(empty)


# A minimal pi `--mode json` stream (as harbor's --no-session pi run emits to pi.txt).
_PI_STREAM_LINES = [
    'Warning: Model "z-ai/glm-5.2" not found for provider "openrouter". Using custom model id.',
    json.dumps({"type": "session", "version": 3, "id": "abc", "cwd": "/app"}),
    json.dumps({"type": "agent_start"}),
    json.dumps({"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "Fix add()"}]}}),
    json.dumps({"type": "message_end", "message": {"role": "assistant", "provider": "openrouter",
        "model": "z-ai/glm-5.2", "content": [{"type": "text", "text": "Fixed."}]}}),
]


def test_pi_stream_to_session_events(tmp_path):
    pi_txt = tmp_path / "pi.txt"
    pi_txt.write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
    events = hb._pi_stream_to_session_events(pi_txt)
    assert [e["type"] for e in events] == ["session", "message", "model_change", "message"]
    mc = next(e for e in events if e["type"] == "model_change")
    assert mc == {"type": "model_change", "provider": "openrouter", "modelId": "z-ai/glm-5.2"}


def test_native_trace_pi_stream(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "pi.txt").write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
    lines, native_dir = hb._native_trace(cfg, agent_dir, "add-bug")
    assert lines and native_dir == tmp_path / "bench" / "sessions" / "add-bug"
    assert (native_dir / "pi.jsonl").is_file()
    types = {json.loads(line)["type"] for line in lines}
    assert "session" in types and "message" in types


def test_rewards_from_result_and_files(tmp_path):
    class R:
        verifier_result = {"rewards": {"reward": 1.0, "sub": 0.5}}
    assert hb._rewards_from_result(R()) == {"reward": 1.0, "sub": 0.5}

    class N:
        verifier_result = None
    assert hb._rewards_from_result(N()) is None

    d = tmp_path / "trial"
    d.mkdir()
    (d / "reward.txt").write_text("0.0\n", encoding="utf-8")
    assert hb._rewards_from_files(d) == {"reward": 0.0}


# ------------------------------------------------------------------- harbor backend run/tasks

def test_harbor_tasks_local(tmp_path):
    pytest.importorskip("harbor")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "add-bug").mkdir(parents=True)
    (tasks_dir / "add-bug" / "task.toml").write_text("", encoding="utf-8")
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source=str(tasks_dir))
    tasks = list(hb.HarborBackend().tasks(cfg, source))
    assert [t.id for t in tasks] == ["add-bug"]


def test_harbor_run_builds_benchrun_from_trial(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "pi.txt").write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")

    class _Trial:
        paths = type("P", (), {"agent_dir": agent_dir})()

    class _Result:
        verifier_result = {"rewards": {"reward": 1.0}}
        exception_info = None

    async def _fake_create_and_run(config):
        return _Trial(), _Result()

    monkeypatch.setattr(hb, "_create_and_run", _fake_create_and_run)
    monkeypatch.setattr(hb, "_build_trial_config", lambda cfg, source, task_dir, trials_dir: object())
    cfg = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source="terminal-bench@2.0")
    run = hb.HarborBackend().run(cfg, source, base.BenchTask(id="add-bug", raw=tmp_path / "task"))
    assert run.rewards == {"reward": 1.0}
    assert run.native_lines and run.metadata.get("exception") is None


# ------------------------------------------------------------------- driver

class _FakeBackend:
    type = "fake"

    def __init__(self, runs):
        self.runs = runs
        self.ran: list[str] = []

    def require(self):
        pass

    def tasks(self, cfg, source, *, refresh=False):
        return [base.BenchTask(id=task_id) for task_id in self.runs]

    def run(self, cfg, source, task):
        self.ran.append(task.id)
        result = self.runs[task.id]
        if isinstance(result, Exception):
            raise result
        return result


def _bench_cfg(tmp_path):
    return Config(
        agent={"provider": "pi"},
        bench={"sources": [{"type": "fake", "source": "S"}]},
        output={"traces_dir": tmp_path / "output"},
    )


def test_run_bench_drives_sources_and_harvests(tmp_path, monkeypatch):
    fake = _FakeBackend({
        "t1": base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 1.0}),
        "t2": base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 0.0}),
    })
    monkeypatch.setattr(bench_runner, "get_backend", lambda t: fake)
    written = run_bench(_bench_cfg(tmp_path))
    assert len(written) == 2 and fake.ran == ["t1", "t2"]
    s = BenchSource(type="fake", source="S")
    assert (tmp_path / "output" / "passed" / f"{base.bench_stem(s, 't1')}.jsonl").is_file()
    assert (tmp_path / "output" / "failed" / f"{base.bench_stem(s, 't2')}.jsonl").is_file()


def test_run_bench_resume_skips_and_failure_continues(tmp_path, monkeypatch):
    s = BenchSource(type="fake", source="S")
    # Pre-harvest t1 so resume skips it; t2 raises (skipped); t3 succeeds.
    done = tmp_path / "output" / "passed" / f"{base.bench_stem(s, 't1')}.jsonl"
    done.parent.mkdir(parents=True)
    done.write_text('{"type":"session"}\n', encoding="utf-8")
    fake = _FakeBackend({
        "t1": base.BenchRun(native_lines=['{"x":1}'], rewards={"reward": 1.0}),
        "t2": RuntimeError("docker boom"),
        "t3": base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 1.0}),
    })
    monkeypatch.setattr(bench_runner, "get_backend", lambda t: fake)
    run_bench(_bench_cfg(tmp_path), resume=True)
    assert "t1" not in fake.ran  # skipped via resume
    assert fake.ran == ["t2", "t3"]
    assert (tmp_path / "output" / "passed" / f"{base.bench_stem(s, 't3')}.jsonl").is_file()


def test_run_bench_unknown_type_aborts(tmp_path):
    # An unregistered source type -> a clear unknown-type error (harbor + swe-bench are known).
    cfg = Config(bench={"sources": [{"type": "nope", "source": "x"}]}, output={"traces_dir": tmp_path / "o"})
    with pytest.raises(RuntimeError, match="Unknown bench source type"):
        run_bench(cfg)


def test_run_bench_honors_max_concurrency(tmp_path, monkeypatch):
    import threading
    import time

    state = {"current": 0, "peak": 0}
    lock = threading.Lock()

    class _ConcBackend:
        type = "fake"

        def require(self):
            pass

        def tasks(self, cfg, source, *, refresh=False):
            return [base.BenchTask(id=f"t{i}") for i in range(6)]

        def run(self, cfg, source, task):
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1
            return base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 1.0})

    monkeypatch.setattr(bench_runner, "get_backend", lambda t: _ConcBackend())
    cfg = Config(
        agent={"provider": "pi"},
        bench={"sources": [{"type": "fake", "source": "S"}]},
        output={"traces_dir": tmp_path / "output"},
        max_concurrency=3,
    )
    written = run_bench(cfg)
    assert len(written) == 6
    assert 1 < state["peak"] <= 3  # tasks ran concurrently, capped at max_concurrency
