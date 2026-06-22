"""Tests for shared Langfuse config + Claude Code and Hermes tracing wiring.

Hermetic: no Docker, network, or real Langfuse.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from teich.config import Config, LangfuseConfig, ModelConfig
from teich.runner import ClaudeCodeRunner, HermesRunner


# -- shared config -----------------------------------------------------------

def test_langfuse_config_alias_is_shared_type():
    # The Codex-era name is kept as an alias of the shared type.
    from teich.config import CodexLangfuseConfig

    assert CodexLangfuseConfig is LangfuseConfig


def test_shared_langfuse_disabled_by_default():
    cfg = Config()
    assert cfg.agent.langfuse.enabled is False
    assert cfg.agent.effective_langfuse.enabled is False


def test_shared_langfuse_requires_all_credentials():
    for missing in ("public_key", "secret_key", "base_url"):
        kwargs = {"public_key": "pk", "secret_key": "sk", "base_url": "https://x"}
        del kwargs[missing]
        with pytest.raises(ValueError, match=missing):
            LangfuseConfig(enabled=True, **kwargs)


def test_effective_langfuse_prefers_codex_override():
    cfg = Config(
        agent={
            "provider": "codex",
            "langfuse": {"enabled": True, "public_key": "pkS", "secret_key": "skS", "base_url": "https://shared"},
            "codex": {"langfuse": {"enabled": True, "public_key": "pkC", "secret_key": "skC", "base_url": "https://codex"}},
        }
    )
    assert cfg.agent.effective_langfuse.base_url == "https://codex"


def test_effective_langfuse_falls_back_to_shared_when_codex_disabled():
    cfg = Config(
        agent={
            "provider": "codex",
            "langfuse": {"enabled": True, "public_key": "pkS", "secret_key": "skS", "base_url": "https://shared"},
        }
    )
    assert cfg.agent.effective_langfuse.enabled is True
    assert cfg.agent.effective_langfuse.base_url == "https://shared"


# -- Claude Code env items ---------------------------------------------------

def _shared_langfuse_config(provider: str) -> Config:
    return Config(
        model=ModelConfig(model="claude-sonnet-4-6"),
        agent={
            "provider": provider,
            "langfuse": {
                "enabled": True,
                "public_key": "pk-lf-1",
                "secret_key": "sk-lf-2",
                "base_url": "https://langfuse.example.com",
            },
        },
    )


def test_claude_langfuse_env_items_when_enabled():
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_shared_langfuse_config("claude-code"))
    items = dict(runner._langfuse_env_items())
    assert items["TRACE_TO_LANGFUSE"] == "true"
    assert items["LANGFUSE_PUBLIC_KEY"] == "pk-lf-1"
    assert items["LANGFUSE_SECRET_KEY"] == "sk-lf-2"
    assert items["LANGFUSE_BASE_URL"] == "https://langfuse.example.com"


def test_claude_langfuse_env_items_empty_when_disabled():
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(Config(model=ModelConfig(model="claude-sonnet-4-6")))
    assert runner._langfuse_env_items() == []


# -- Hermes env items --------------------------------------------------------

def test_hermes_langfuse_env_items_when_enabled():
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(_shared_langfuse_config("hermes"))
    items = dict(runner._langfuse_env_items())
    assert items["HERMES_LANGFUSE_PUBLIC_KEY"] == "pk-lf-1"
    assert items["HERMES_LANGFUSE_SECRET_KEY"] == "sk-lf-2"
    assert items["HERMES_LANGFUSE_BASE_URL"] == "https://langfuse.example.com"


def test_hermes_langfuse_env_items_empty_when_disabled():
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(Config(model=ModelConfig(model="hermes-4")))
    assert runner._langfuse_env_items() == []


# -- Claude settings.json hook -----------------------------------------------

def test_claude_prepare_home_writes_stop_hook(tmp_path: Path):
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(_shared_langfuse_config("claude-code"))
    home = tmp_path / "home"
    home.mkdir()
    runner._prepare_agent_home(home)
    import json as _json
    settings = _json.loads((home / "settings.json").read_text())
    cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    # Must use the venv python by absolute path (claude sanitizes PATH for hooks).
    assert cmd.startswith("/opt/venv/bin/python3 ")
    assert cmd.endswith("langfuse_hook.py")


def test_claude_prepare_home_noop_when_disabled(tmp_path: Path):
    with patch.object(ClaudeCodeRunner, "_ensure_image"):
        runner = ClaudeCodeRunner(Config(model=ModelConfig(model="claude-sonnet-4-6")))
    home = tmp_path / "home"
    home.mkdir()
    runner._prepare_agent_home(home)
    assert not (home / "settings.json").exists()


# -- Hermes plugin config ----------------------------------------------------

def test_hermes_prepare_home_writes_plugin_config(tmp_path: Path):
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(_shared_langfuse_config("hermes"))
    home = tmp_path / "home"
    home.mkdir()
    runner._prepare_agent_home(home)
    text = (home / "config.yaml").read_text()
    assert "observability/langfuse" in text
    assert "plugins" in text


def test_hermes_prepare_home_noop_when_disabled(tmp_path: Path):
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(Config(model=ModelConfig(model="hermes-4")))
    home = tmp_path / "home"
    home.mkdir()
    runner._prepare_agent_home(home)
    assert not (home / "config.yaml").exists()


def test_hermes_shell_drops_ignore_user_config_when_langfuse_enabled():
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(_shared_langfuse_config("hermes"))
    shell = runner._build_shell_command()
    assert "--ignore-user-config" not in shell


def test_hermes_shell_keeps_ignore_user_config_when_disabled():
    with patch.object(HermesRunner, "_ensure_image"):
        runner = HermesRunner(Config(model=ModelConfig(model="hermes-4")))
    shell = runner._build_shell_command()
    assert "--ignore-user-config" in shell
