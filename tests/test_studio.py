"""Tests for the Teich Studio backend (project state, events, API)."""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from teich.cli import _configure_studio_event_loop_policy
from teich.studio.events import summarize_chat_row, summarize_event, summarize_trace_events
from teich.studio.project import ProjectState
from teich.studio.server import _settle_terminal_tasks, create_app, detect_trace_provider


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as test_client:
        test_client.project_dir = tmp_path
        yield test_client


# ---------------------------------------------------------------------------
# Project state
# ---------------------------------------------------------------------------

def test_ensure_initialized_creates_files(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "prompts.jsonl").exists()


def test_write_config_merges_and_validates(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    merged = state.write_config_data({"model": {"model": "test/model"}, "max_concurrency": 4})
    assert merged["model"]["model"] == "test/model"
    assert merged["max_concurrency"] == 4
    on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert on_disk["model"]["model"] == "test/model"
    # untouched section preserved
    assert on_disk["agent"]["provider"]


def test_write_config_rejects_invalid(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    with pytest.raises(Exception):
        state.write_config_data({"max_concurrency": 0})


def test_prompts_round_trip(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    rows = [
        {"prompt": "Build a CLI"},
        {"prompt": "Fix the bug", "system": "Be terse", "follow_up_prompts": ["Add tests"]},
    ]
    state.write_prompts(rows)
    loaded = state.read_prompts()
    assert loaded[0] == {"prompt": "Build a CLI"}
    assert loaded[1]["follow_up_prompts"] == ["Add tests"]


def test_read_prompts_supports_csv_prompt_file(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    (tmp_path / "prompts.csv").write_text(
        "prompt,system,github_repo\n"
        "Build the feature,Be concise,owner/repo\n",
        encoding="utf-8",
    )
    state.write_config_data({"prompts_file": "prompts.csv"})

    assert state.read_prompts() == [
        {"prompt": "Build the feature", "system": "Be concise", "github_repo": "owner/repo"}
    ]


def test_write_prompts_migrates_non_jsonl_prompt_file(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    (tmp_path / "prompts.csv").write_text("prompt\nold prompt\n", encoding="utf-8")
    state.write_config_data({"prompts_file": "prompts.csv"})

    path = state.write_prompts([{"prompt": "new prompt"}])

    assert path == tmp_path / "prompts.jsonl"
    assert json.loads(path.read_text(encoding="utf-8").strip()) == {"prompt": "new prompt"}
    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert config["prompts_file"] == "prompts.jsonl"


def test_import_prompts_append_and_replace(tmp_path):
    state = ProjectState(tmp_path)
    state.ensure_initialized()
    state.write_prompts([{"prompt": "one"}])
    state.import_prompts_text('{"prompt": "two"}\n"three"\n', replace=False)
    assert [row["prompt"] for row in state.read_prompts()] == ["one", "two", "three"]
    state.import_prompts_text('{"prompt": "only"}', replace=True)
    assert [row["prompt"] for row in state.read_prompts()] == ["only"]


# ---------------------------------------------------------------------------
# Event summarizers
# ---------------------------------------------------------------------------

def test_summarize_codex_events():
    assistant = {
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
    }
    tool = {
        "type": "response_item",
        "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"},
    }
    assert summarize_event("codex", assistant)[0]["kind"] == "assistant"
    tool_events = summarize_event("codex", tool)
    assert tool_events[0]["kind"] == "tool_call"
    assert tool_events[0]["name"] == "exec_command"


def test_summarize_claude_stream_json():
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "done"},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ]
        },
    }
    kinds = [e["kind"] for e in summarize_event("claude-code", event)]
    assert kinds == ["thinking", "assistant", "tool_call"]


def test_summarize_hermes_external():
    event = {
        "type": "external_message",
        "role": "assistant",
        "content": "answer",
        "tool_calls": [{"function": {"name": "skill_manage", "arguments": "{}"}}],
    }
    kinds = [e["kind"] for e in summarize_event("hermes", event)]
    assert kinds == ["tool_call", "assistant"]


def test_summarize_trace_events_includes_user_turns():
    events = [
        {"type": "external_session_meta", "payload": {}},
        {"type": "external_message", "role": "user", "content": "question"},
        {"type": "external_message", "role": "assistant", "content": "answer"},
    ]
    display = summarize_trace_events("hermes", events)
    assert [e["kind"] for e in display] == ["user", "assistant"]


def test_summarize_chat_row():
    row = {
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "thinking": "greeting"},
        ]
    }
    kinds = [e["kind"] for e in summarize_chat_row(row)]
    assert kinds == ["system", "user", "thinking", "assistant"]


def test_detect_trace_provider():
    assert detect_trace_provider([{"type": "session_meta", "payload": {}}]) == "codex"
    assert detect_trace_provider([{"type": "session", "id": "x"}]) == "pi"
    assert detect_trace_provider([{"type": "external_session_meta"}]) == "hermes"
    assert detect_trace_provider([{"type": "user", "sessionId": "abc"}]) == "claude-code"
    assert detect_trace_provider([{"messages": []}]) == "chat"


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

def test_status_endpoint(client):
    payload = client.get("/api/status").json()
    assert payload["config_exists"] is True
    assert payload["prompts_count"] == 0
    assert {p["id"] for p in payload["providers"]} == {"pi", "codex", "claude-code", "hermes", "chat"}


def test_config_endpoints(client):
    config = client.get("/api/config").json()["config"]
    assert config["agent"]["provider"]
    response = client.put("/api/config", json={"config": {"model": {"model": "acme/model-1"}}})
    assert response.status_code == 200
    assert response.json()["config"]["model"]["model"] == "acme/model-1"
    bad = client.put("/api/config", json={"config": {"max_concurrency": -1}})
    assert bad.status_code == 400


def test_config_rejects_chat_with_direct_anthropic_api(client):
    response = client.put(
        "/api/config",
        json={
            "config": {
                "agent": {"provider": "chat"},
                "api": {"provider": "anthropic", "base_url": "https://api.anthropic.com"},
            }
        },
    )

    assert response.status_code == 400
    assert "OpenAI-compatible" in response.json()["detail"]
    assert client.get("/api/config").json()["config"]["agent"]["provider"] != "chat"

    custom_direct = client.put(
        "/api/config",
        json={
            "config": {
                "agent": {"provider": "chat"},
                "api": {"provider": "custom", "base_url": "https://api.anthropic.com/"},
            }
        },
    )
    assert custom_direct.status_code == 400
    assert "OpenAI-compatible" in custom_direct.json()["detail"]


def test_session_override_rejects_chat_with_direct_anthropic_api(client):
    response = client.put(
        "/api/config",
        json={
            "config": {
                "agent": {"provider": "claude-code"},
                "api": {"provider": "anthropic", "base_url": "https://api.anthropic.com"},
            }
        },
    )
    assert response.status_code == 200

    session = client.post("/api/sessions", json={"provider": "chat"})

    assert session.status_code == 400
    assert "OpenAI-compatible" in session.json()["detail"]


def test_prompts_endpoints(client):
    response = client.put(
        "/api/prompts",
        json={"prompts": [{"prompt": "hello world", "follow_up_prompts": ["again"]}]},
    )
    assert response.status_code == 200
    prompts = client.get("/api/prompts").json()["prompts"]
    assert prompts[0]["prompt"] == "hello world"

    imported = client.post(
        "/api/prompts/import",
        json={"text": '{"prompt": "uploaded"}', "replace": False},
    )
    assert imported.status_code == 200
    assert len(imported.json()["prompts"]) == 2

    invalid = client.post("/api/prompts/import", json={"text": "{not json", "replace": False})
    assert invalid.status_code == 400


def test_trace_listing_and_preview(client):
    output_dir = client.project_dir / "output"
    output_dir.mkdir()
    trace = output_dir / "hermes-agent-test.jsonl"
    events = [
        {"type": "external_session_meta", "payload": {"id": "x"}},
        {"type": "external_message", "role": "user", "content": "do the thing"},
        {"type": "external_message", "role": "assistant", "content": "did the thing"},
    ]
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (output_dir / "failures").mkdir()
    (output_dir / "failures" / "bad.jsonl").write_text("{}\n", encoding="utf-8")

    listing = client.get("/api/traces").json()["traces"]
    assert [t["name"] for t in listing] == ["hermes-agent-test.jsonl"]

    preview = client.get("/api/traces/preview", params={"name": "hermes-agent-test.jsonl"}).json()
    assert preview["provider"] == "hermes"
    assert [e["kind"] for e in preview["display"]] == ["user", "assistant"]

    missing = client.get("/api/traces/preview", params={"name": "nope.jsonl"})
    assert missing.status_code == 404
    escape = client.get("/api/traces/preview", params={"name": "../config.yaml"})
    assert escape.status_code in {400, 404}


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Teich Studio" in response.text


def test_session_endpoints_validation(client):
    missing = client.get("/api/sessions/does-not-exist")
    assert missing.status_code == 404
    assert client.get("/api/sessions").json()["sessions"] == []


def test_studio_uses_selector_event_loop_policy_on_windows():
    class DummyPolicy:
        pass

    class DummyAsyncio:
        WindowsSelectorEventLoopPolicy = DummyPolicy

        def __init__(self) -> None:
            self.policy = None

        def set_event_loop_policy(self, policy) -> None:
            self.policy = policy

    asyncio_module = DummyAsyncio()

    assert _configure_studio_event_loop_policy(platform="win32", asyncio_module=asyncio_module)
    assert isinstance(asyncio_module.policy, DummyPolicy)


def test_studio_event_loop_policy_noop_off_windows():
    class DummyAsyncio:
        def set_event_loop_policy(self, policy) -> None:
            raise AssertionError("event loop policy should not be changed off Windows")

    assert not _configure_studio_event_loop_policy(platform="linux", asyncio_module=DummyAsyncio())


def test_terminal_task_cleanup_suppresses_disconnects():
    async def run() -> None:
        async def disconnect() -> None:
            raise WebSocketDisconnect(code=1005)

        async def wait_forever() -> None:
            await asyncio.sleep(60)

        done_task = asyncio.create_task(disconnect())
        pending_task = asyncio.create_task(wait_forever())
        await asyncio.sleep(0)

        await _settle_terminal_tasks({done_task}, {pending_task})

        assert pending_task.cancelled()

    asyncio.run(run())


def test_terminal_task_cleanup_propagates_unexpected_errors():
    async def run() -> None:
        async def fail() -> None:
            raise ValueError("boom")

        async def wait_forever() -> None:
            await asyncio.sleep(60)

        done_task = asyncio.create_task(fail())
        pending_task = asyncio.create_task(wait_forever())
        await asyncio.sleep(0)

        with pytest.raises(ValueError, match="boom"):
            await _settle_terminal_tasks({done_task}, {pending_task})

        assert pending_task.cancelled()

    asyncio.run(run())
