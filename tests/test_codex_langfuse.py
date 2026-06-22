"""Tests for Codex -> Langfuse tracing wiring.

These tests are hermetic: they never touch Docker, the network, or a real
Langfuse instance. They cover config validation, the config.toml blocks Teich
writes to enable the plugin, the Langfuse env vars passed to the container, and
the per-session install of the (image-baked) plugin tree into CODEX_HOME.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from teich.config import CodexLangfuseConfig, Config, ModelConfig
from teich.runner import CodexRunner


def _langfuse_config(**overrides) -> Config:
    langfuse = {
        "enabled": True,
        "public_key": "pk-lf-123",
        "secret_key": "sk-lf-456",
        "base_url": "https://langfuse.example.com",
    }
    langfuse.update(overrides)
    return Config(
        model=ModelConfig(model="gpt-5.5"),
        agent={"provider": "codex", "codex": {"langfuse": langfuse}},
    )


# -- config validation -------------------------------------------------------

def test_langfuse_disabled_by_default():
    cfg = Config()
    assert cfg.agent.codex.langfuse.enabled is False


def test_langfuse_enabled_requires_all_credentials():
    with pytest.raises(ValueError, match="public_key"):
        CodexLangfuseConfig(enabled=True, secret_key="sk", base_url="https://x")
    with pytest.raises(ValueError, match="secret_key"):
        CodexLangfuseConfig(enabled=True, public_key="pk", base_url="https://x")
    with pytest.raises(ValueError, match="base_url"):
        CodexLangfuseConfig(enabled=True, public_key="pk", secret_key="sk")


def test_langfuse_enabled_with_all_credentials_ok():
    cfg = CodexLangfuseConfig(
        enabled=True, public_key="pk", secret_key="sk", base_url="https://x"
    )
    assert cfg.enabled and cfg.public_key == "pk"


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("public_key", {"public_key": "   ", "secret_key": "sk", "base_url": "https://x"}),
        ("secret_key", {"public_key": "pk", "secret_key": "   ", "base_url": "https://x"}),
        ("base_url", {"public_key": "pk", "secret_key": "sk", "base_url": "   "}),
    ],
)
def test_langfuse_blank_credential_is_rejected(field: str, kwargs: dict[str, str]):
    with pytest.raises(ValueError, match=field):
        CodexLangfuseConfig(enabled=True, **kwargs)


# -- config.toml blocks ------------------------------------------------------

def test_codex_config_writes_langfuse_blocks_when_enabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "[features]" in content
    assert "plugin_hooks = true" in content
    assert '[plugins."tracing@codex-observability-plugin"]' in content
    assert "enabled = true" in content


def test_codex_config_omits_langfuse_blocks_when_disabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    codex_home = tmp_path / ".codex"
    runner._write_codex_config(codex_home)
    content = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "plugin_hooks" not in content
    assert "tracing@codex-observability-plugin" not in content


# -- container env vars ------------------------------------------------------

def test_codex_command_passes_langfuse_env_when_enabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert "TRACE_TO_LANGFUSE=true" in cmd
    assert "LANGFUSE_PUBLIC_KEY=pk-lf-123" in cmd
    assert "LANGFUSE_SECRET_KEY=sk-lf-456" in cmd
    assert "LANGFUSE_BASE_URL=https://langfuse.example.com" in cmd


def test_codex_command_omits_langfuse_env_when_disabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert not any(part.startswith("TRACE_TO_LANGFUSE=") for part in cmd)
    assert not any(part.startswith("LANGFUSE_") for part in cmd)


def test_codex_command_rewrites_localhost_langfuse_base_url(tmp_path: Path):
    cfg = _langfuse_config(base_url="http://localhost:3000")
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(cfg)
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert "LANGFUSE_BASE_URL=http://host.docker.internal:3000" in cmd
    assert "host.docker.internal:host-gateway" in cmd


def test_codex_command_adds_host_gateway_for_host_local_langfuse(tmp_path: Path):
    cfg = _langfuse_config(base_url="http://host.docker.internal:3000")
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(cfg)
    cmd = runner._build_codex_command(
        "Build app",
        workspace=tmp_path / "ws",
        codex_home=tmp_path / "ch",
        container_name="teich-codex-x",
    )
    assert "host.docker.internal:host-gateway" in cmd


# -- hook-trust bypass (exec) ------------------------------------------------

def test_codex_agent_command_bypasses_hook_trust_when_enabled():
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    cmd = runner._build_codex_agent_command()
    assert "exec" in cmd
    assert "--dangerously-bypass-hook-trust" in cmd


def test_codex_agent_command_no_hook_trust_bypass_when_disabled():
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    cmd = runner._build_codex_agent_command()
    assert "--dangerously-bypass-hook-trust" not in cmd


# -- per-session plugin install ---------------------------------------------

def test_install_codex_langfuse_plugin_copies_tree(tmp_path: Path):
    # Fake the image-baked cache so the test never touches Docker.
    cache = tmp_path / "cache"
    leaf = cache / "plugins" / "cache" / "codex-observability-plugin" / "tracing" / "0.1.0" / "dist"
    leaf.mkdir(parents=True)
    (leaf / "index.mjs").write_text("// bundle", encoding="utf-8")

    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(_langfuse_config())
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()

    with patch.object(runner, "_ensure_langfuse_plugin_cache", return_value=cache):
        runner._install_codex_langfuse_plugin(codex_home)

    installed = (
        codex_home / "plugins" / "cache" / "codex-observability-plugin"
        / "tracing" / "0.1.0" / "dist" / "index.mjs"
    )
    assert installed.exists()
    assert installed.read_text(encoding="utf-8") == "// bundle"


def test_install_codex_langfuse_plugin_noop_when_disabled(tmp_path: Path):
    with patch.object(CodexRunner, "_ensure_image"):
        runner = CodexRunner(Config(model=ModelConfig(model="gpt-5.5")))
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    # Disabled -> cache is None -> nothing copied, no error.
    runner._install_codex_langfuse_plugin(codex_home)
    assert not (codex_home / "plugins").exists()
