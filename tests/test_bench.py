"""Tests for bench-mode config + guards (the harbor driver itself is Docker/integration)."""

import pytest

from teich.bench import run_bench
import teich.bench.runner as bench_runner
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


def test_run_bench_validates_then_reaches_driver(monkeypatch):
    # Stub the harbor requirement so the test doesn't depend on the optional extra.
    monkeypatch.setattr(bench_runner, "_require_harbor", lambda: object())
    cfg = Config(bench={"source": "./tasks"})
    with pytest.raises(RuntimeError, match="not wired yet"):
        run_bench(cfg)
