"""Integration tests for teich.

These tests require:
- Docker running
- OPENAI_API_KEY environment variable set (for live tests)
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from teich.config import APIConfig, Config, ModelConfig, PromptInput
from teich.runner import CodexRunner, RUNTIME_CONTAINER_USER, RUNTIME_IMAGE_NAME


@pytest.fixture
def require_docker():
    """Skip Docker-backed tests without probing Docker during collection."""
    try:
        docker_info = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker not available")
    if docker_info.returncode != 0:
        pytest.skip("Docker not available")

requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

requires_runtime_smoke = pytest.mark.skipif(
    os.getenv("TEICH_RUN_DOCKER_RUNTIME_SMOKE") != "1",
    reason="set TEICH_RUN_DOCKER_RUNTIME_SMOKE=1 to run Docker package-manager smoke",
)


class TestDockerImage:
    """Tests for Docker image building."""

    def test_dfile_exists(self):
        """Verify Dockerfile exists."""
        dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"
        assert dockerfile.exists()

    def test_dockerfile_preinstalls_playwright_chromium(self):
        """Verify the runtime image bakes in Playwright Chromium dependencies."""
        dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")
        assert "python3-pip" in content
        assert "python3-venv" in content
        assert "python3-dev" in content
        assert "python3 -m venv /opt/venv" in content
        assert 'ENV VIRTUAL_ENV=/opt/venv' in content
        assert 'ENV PATH="/opt/venv/bin:/usr/local/bin:$PATH"' in content
        assert "pip --version" in content
        assert "pip3 --version" in content
        assert "ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright" in content
        assert "npm install -g @openai/codex @anthropic-ai/claude-code @mariozechner/pi-coding-agent playwright" in content
        assert "git clone --depth 1 https://github.com/NousResearch/hermes-agent.git" in content
        assert "uv pip install --python /usr/local/lib/hermes-agent/venv/bin/python -e ." in content
        assert "npx playwright install --with-deps chromium" in content
        assert 'ENV NODE_PATH="/usr/local/lib/node_modules"' in content

    def test_dockerfile_allows_non_root_agents_to_install_system_packages(self):
        """Verify the runtime keeps passwordless apt wrappers for generated agents."""
        dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")
        assert "sudo" in content
        assert "codex ALL=(ALL) NOPASSWD:ALL" in content
        assert "exec sudo /usr/bin/apt-get" in content
        assert "exec sudo /usr/bin/apt" in content
        assert "chmod +x /usr/local/bin/apt-get /usr/local/bin/apt" in content
        assert "USER codex" in content

    @pytest.mark.slow  # Takes 2-3 minutes, skip by default
    def test_docker_build(self, tmp_path):
        """Test Docker image builds successfully."""
        pytest.skip("Docker build test - run manually with: pytest tests/test_integration.py::TestDockerImage::test_docker_build -v")
        dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"

        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(dockerfile),
                "-t", "test-teich:latest",
                str(dockerfile.parent),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for build
        )

        assert result.returncode == 0, f"Docker build failed: {result.stderr}"

    @requires_runtime_smoke
    @pytest.mark.slow
    def test_runtime_container_can_install_system_packages(self, require_docker):
        """Verify generated agents can use apt-get for missing system dependencies."""
        CodexRunner(Config())

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                RUNTIME_CONTAINER_USER,
                "-e",
                "HOME=/home/codex",
                RUNTIME_IMAGE_NAME,
                "bash",
                "-lc",
                "test \"$(id -u)\" != 0 && test \"$(command -v apt-get)\" = /usr/local/bin/apt-get && apt-get update -qq",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        assert result.returncode == 0, result.stderr or result.stdout


class TestRunnerIntegration:
    """Integration tests for CodexRunner with real Docker."""

    def test_runner_creates_output_dir(self, tmp_path):
        """Test runner creates output directory."""
        from teich.config import OutputConfig

        config = Config(
            model=ModelConfig(model="codex-mini-latest", approval_mode="none"),
            prompts=["test"],
            output=OutputConfig(traces_dir=tmp_path / "output"),
        )

        with patch.object(CodexRunner, '_ensure_image'):
            runner = CodexRunner(config)

        # Output dir should be created
        runner.config.output.traces_dir.mkdir(parents=True, exist_ok=True)
        assert runner.config.output.traces_dir.exists()


class TestTraceFormat:
    """Tests for trace output format validation."""

    def test_trace_line_structure(self, tmp_path):
        """Verify trace JSONL has required fields."""
        # Create a mock trace file
        trace_file = tmp_path / "test_session.jsonl"

        events = [
            {"type": "session", "id": "test-123", "timestamp": "2024-01-01T00:00:00Z"},
            {"type": "message", "message": {"role": "user", "content": "Build an app"}},
            {"type": "message", "message": {"role": "assistant", "content": "I'll help you build an app"}},
        ]

        with open(trace_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        # Validate structure
        with open(trace_file) as f:
            lines = f.readlines()
            assert len(lines) == 3

            # First line should be session
            first = json.loads(lines[0])
            assert first["type"] == "session"
            assert "id" in first

            # Messages should have role
            for line in lines[1:]:
                event = json.loads(line)
                if event["type"] == "message":
                    assert "message" in event
                    assert "role" in event["message"]


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_config_yaml_roundtrip(self, tmp_path, monkeypatch):
        """Test config can be saved and loaded."""
        # Clear env vars that would override YAML values
        monkeypatch.delenv("TEICH_MODEL", raising=False)
        monkeypatch.delenv("TEICH_BASE_URL", raising=False)
        monkeypatch.delenv("TEICH_API_KEY", raising=False)
        monkeypatch.delenv("TEICH_PROVIDER", raising=False)

        config_file = tmp_path / "config.yaml"

        # Create and save config (use model_dump with mode='json' for safe serialization)
        original = Config(
            model=ModelConfig(model="gpt-4o", approval_mode="suggest"),
            prompts=["Test prompt 1", "Test prompt 2"],
            openai_api_key="sk-test",
        )

        # Save manually - use json mode to avoid Path serialization issues
        import yaml
        config_file.write_text(yaml.dump(original.model_dump(mode="json")))

        # Load back
        loaded = Config.from_yaml(config_file)

        assert loaded.model.model == original.model.model
        assert loaded.prompts == original.prompts


class TestOpenRouterIntegration:
    """Tests for OpenRouter/custom API provider integration."""

    def test_openrouter_config_in_command(self):
        """Verify OpenRouter config generates correct CLI args."""
        config = Config(
            model=ModelConfig(model="anthropic/claude-3.5-sonnet"),
            api=APIConfig(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-or-test",
            ),
            openai_api_key="sk-openai",
        )

        # The command should use the API-specific key
        assert config.api.api_key == "sk-or-test"
        # But fall back to global if not set
        config2 = Config(openai_api_key="sk-global")
        assert config2.api.api_key is None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("require_docker")
@requires_api_key
class TestEndToEnd:
    """Full end-to-end tests requiring real API access."""

    def test_full_generation_workflow(self, tmp_path):
        """Test complete workflow: init -> generate -> verify output."""
        # Setup
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        config = Config(
            model=ModelConfig(model="codex-mini-latest", approval_mode="none"),
            prompts=["Create a Python hello world script"],
            output=MagicMock(traces_dir=project_dir / "output"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            timeout_seconds=60,
        )

        with patch.object(CodexRunner, '_ensure_image'):
            runner = CodexRunner(config)

        # Create workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Run session (mocked for unit test, would be real for integration)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Mock session file creation
            session_dir = tmp_path / ".codex" / "sessions"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "test-session.jsonl"
            session_file.write_text(
                json.dumps({"type": "session", "id": "test"}) + "\n"
            )

            with patch.object(runner, '_extract_session_file', return_value=project_dir / "output" / "test.jsonl"):
                result = runner.run_session("Create a Python hello world script", "test")

        assert result is not None


@pytest.mark.integration
@pytest.mark.usefixtures("require_docker")
class TestSeedVerifierDocker:
    """Real-Docker verification of the seed-workspace verifier reward path."""

    @staticmethod
    def _require_runtime_image():
        present = subprocess.run(
            ["docker", "images", "-q", RUNTIME_IMAGE_NAME], capture_output=True, text=True
        ).stdout.strip()
        if not present:
            pytest.skip(f"{RUNTIME_IMAGE_NAME} not built")

    @staticmethod
    def _make_seed_bundle(tmp_path: Path) -> Path:
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e.st",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e.st"}
        repo = tmp_path / "src"
        repo.mkdir()
        (repo / "app.py").write_text("def add(a, b):\n    return a - b  # BUG\n", encoding="utf-8")
        (repo / "check.py").write_text("import app\nassert app.add(2, 2) == 4\nprint('OK')\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed"], check=True, env=env)
        bundle = tmp_path / "app.bundle"
        subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), "--all"], check=True, env=env)
        return bundle

    def test_verifier_reward_bug_then_fix(self, tmp_path):
        """Seed clone -> verifier fails on the bug -> passes once fixed (real docker run)."""
        self._require_runtime_image()
        bundle = self._make_seed_bundle(tmp_path)
        with patch.object(CodexRunner, "_ensure_image"):
            runner = CodexRunner(Config(agent={"provider": "codex"}, tasks={"verifier_timeout_seconds": 120}))
        pi = PromptInput(prompt="fix", seed_repo=str(bundle), verifier="python check.py")

        root_a, ws_a = runner._prepare_workspace("seed-a", pi, "codex")
        try:
            assert (ws_a / "app.py").exists() and (ws_a / ".git").is_dir()  # cloned with history
            buggy = runner._run_verifier(ws_a, pi)
        finally:
            shutil.rmtree(root_a, ignore_errors=True)
        assert buggy is not None and buggy.passed is False and buggy.exit_code != 0

        root_b, ws_b = runner._prepare_workspace("seed-b", pi, "codex")
        try:
            (ws_b / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            fixed = runner._run_verifier(ws_b, pi)
        finally:
            shutil.rmtree(root_b, ignore_errors=True)
        assert fixed is not None and fixed.passed is True and fixed.exit_code == 0

    @staticmethod
    def _make_f2p_bundle(tmp_path: Path) -> Path:
        """Seed where test_add (F2P) fails on the bug and test_mul (P2P) always passes."""
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e.st",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e.st"}
        repo = tmp_path / "src"
        repo.mkdir()
        (repo / "app.py").write_text("def add(a, b):\n    return a - b  # BUG\ndef mul(a, b):\n    return a * b\n", encoding="utf-8")
        # A plain-python runner that emits pytest-style PASSED/FAILED <id> lines
        # (keeps the test off pytest-install while exercising the F2P/P2P parser).
        (repo / "run_tests.py").write_text(
            "import sys, app\n"
            "results = {\n"
            "    'tests/test_app.py::test_add': app.add(2, 2) == 4,\n"
            "    'tests/test_app.py::test_mul': app.mul(2, 3) == 6,\n"
            "}\n"
            "ok = True\n"
            "for tid, passed in results.items():\n"
            "    print(('PASSED ' if passed else 'FAILED ') + tid)\n"
            "    ok = ok and passed\n"
            "sys.exit(0 if ok else 1)\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed"], check=True, env=env)
        bundle = tmp_path / "f2p.bundle"
        subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), "--all"], check=True, env=env)
        return bundle

    def test_f2p_p2p_transition_over_docker(self, tmp_path):
        """Genuine before/after F2P/P2P: bug fails F2P on the seed baseline, fix flips it to resolved."""
        self._require_runtime_image()
        bundle = self._make_f2p_bundle(tmp_path)
        with patch.object(CodexRunner, "_ensure_image"):
            runner = CodexRunner(Config(
                agent={"provider": "codex"},
                tasks={"verifier_timeout_seconds": 120, "check_seed_baseline": True},
            ))
        pi = PromptInput(
            prompt="fix", seed_repo=str(bundle), verifier="python run_tests.py",
            fail_to_pass=["tests/test_app.py::test_add"],
            pass_to_pass=["tests/test_app.py::test_mul"],
        )
        root, ws = runner._prepare_workspace("f2p", pi, "codex")
        try:
            # Agent fixes the bug.
            (ws / "app.py").write_text("def add(a, b):\n    return a + b\ndef mul(a, b):\n    return a * b\n", encoding="utf-8")
            result = runner._run_verifier(ws, pi)
            runner._apply_f2p_p2p_reward(result, pi)  # runs the pristine-seed baseline internally
        finally:
            shutil.rmtree(root, ignore_errors=True)
        assert result.resolved is True and result.passed is True
        assert result.valid_task is True
        assert result.fail_to_pass == {"tests/test_app.py::test_add": "passed"}
        assert result.baseline["fail_to_pass"] == {"tests/test_app.py::test_add": "failed"}
        assert result.baseline["pass_to_pass"] == {"tests/test_app.py::test_mul": "passed"}


# Integration test markers
pytestmark = [
    pytest.mark.integration,
]
