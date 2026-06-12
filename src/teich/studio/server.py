"""FastAPI server for Teich Studio."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import asyncio

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .events import summarize_chat_row, summarize_trace_events
from .generation import GenerationManager
from .interactive import EventLog, SessionManager
from .project import ProjectState, validate_chat_api_compatibility

STATIC_DIR = Path(__file__).parent / "static"

PROVIDERS = [
    {
        "id": "pi",
        "label": "Pi",
        "kind": "agent",
        "description": "Pi coding agent running in Docker. Great default for coding traces.",
    },
    {
        "id": "codex",
        "label": "Codex",
        "kind": "agent",
        "description": "OpenAI Codex CLI running in Docker.",
    },
    {
        "id": "claude-code",
        "label": "Claude Code",
        "kind": "agent",
        "description": "Anthropic Claude Code CLI running in Docker.",
    },
    {
        "id": "hermes",
        "label": "Hermes",
        "kind": "agent",
        "description": "Nous Hermes Agent CLI running in Docker.",
    },
    {
        "id": "chat",
        "label": "Chat",
        "kind": "chat",
        "description": "Text-only chat distillation via an OpenAI-compatible API. No Docker needed.",
    },
]


class ConfigUpdate(BaseModel):
    config: dict[str, Any]


class PromptsUpdate(BaseModel):
    prompts: list[dict[str, Any]]


class PromptsImport(BaseModel):
    text: str
    replace: bool = False


class GenerateRequest(BaseModel):
    resume: bool = False


class SessionCreate(BaseModel):
    provider: str | None = None
    model: str | None = None
    github_repo: str | None = None
    system: str | None = None


class SessionMessage(BaseModel):
    text: str = Field(min_length=1)


_docker_cache: dict[str, Any] = {"checked_at": 0.0, "available": False, "detail": None}
TERMINAL_READY_STATUSES = {"ready", "live", "exited", "error"}
TERMINAL_STARTUP_NOTICE_SECONDS = 15.0


async def _settle_terminal_tasks(
    done: set[asyncio.Task[None]], pending: set[asyncio.Task[None]]
) -> None:
    """Cancel and await terminal websocket pump tasks after either side exits."""
    for task in pending:
        task.cancel()

    unexpected: BaseException | None = None
    for task in [*done, *pending]:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            if unexpected is None:
                unexpected = exc

    if unexpected is not None:
        raise unexpected


async def _wait_for_terminal_session_ready(
    websocket: WebSocket,
    session: Any,
    *,
    notice_seconds: float = TERMINAL_STARTUP_NOTICE_SECONDS,
    sleep_seconds: float = 0.5,
) -> None:
    """Wait until the session can attach a terminal, without timing out cold Docker builds."""
    next_notice = time.monotonic() + max(notice_seconds, 0.0)
    while session.status not in TERMINAL_READY_STATUSES:
        now = time.monotonic()
        if now >= next_notice:
            await websocket.send_json({
                "type": "status",
                "detail": "Still waiting for the Docker runtime to finish starting...",
            })
            next_notice = now + max(notice_seconds, sleep_seconds)
        await asyncio.sleep(sleep_seconds)


def _docker_status() -> dict[str, Any]:
    now = time.time()
    if now - _docker_cache["checked_at"] < 30:
        return {"available": _docker_cache["available"], "detail": _docker_cache["detail"]}
    available = False
    detail: str | None = None
    if shutil.which("docker") is None:
        detail = "Docker CLI not found"
    else:
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                timeout=8,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0 and result.stdout.strip():
                available = True
                detail = result.stdout.strip()
            else:
                detail = (result.stderr or "").strip() or "Docker daemon not responding"
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)
    _docker_cache.update({"checked_at": now, "available": available, "detail": detail})
    return {"available": available, "detail": detail}


def detect_trace_provider(events: list[dict[str, Any]]) -> str:
    for event in events[:5]:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session_meta":
            return "codex"
        if event_type == "session":
            return "pi"
        if event_type == "external_session_meta":
            return "hermes"
        if event_type in {"response_item", "event_msg"}:
            return "codex"
        if event_type == "message" and isinstance(event.get("message"), dict):
            return "pi"
        if "sessionId" in event or event_type == "queue-operation":
            return "claude-code"
        if isinstance(event.get("messages"), list):
            return "chat"
    return "unknown"


def _sse(log: EventLog, after: int) -> StreamingResponse:
    def stream():
        index = max(after, 0)
        idle_cycles = 0
        while True:
            events = log.wait_for(index, timeout=15.0)
            if events:
                for event in events:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                index += len(events)
                idle_cycles = 0
            else:
                if log.closed:
                    yield "event: end\ndata: {}\n\n"
                    return
                idle_cycles += 1
                yield ": keepalive\n\n"
                if idle_cycles > 240:  # ~1 hour of silence
                    return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(project_dir: Path) -> FastAPI:
    state = ProjectState(project_dir)
    state.ensure_initialized()
    sessions = SessionManager()
    generation = GenerationManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            sessions.shutdown()
            generation.shutdown()

    app = FastAPI(title="Teich Studio", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.project = state
    app.state.sessions = sessions
    app.state.generation = generation

    # ------------------------------------------------------------------
    # Status / config
    # ------------------------------------------------------------------

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        config_error: str | None = None
        api_key_present = False
        provider = None
        model = None
        prompts_count = 0
        try:
            cfg = state.load_config()
            api_key_present = bool(cfg.get_api_key())
            provider = cfg.get_agent_provider()
            model = cfg.get_effective_model()
            prompts_count = len(cfg.get_prompt_inputs())
        except Exception as exc:
            config_error = str(exc)
            try:
                prompts_count = len(state.read_prompts())
            except ValueError:
                prompts_count = -1
        return {
            "project_dir": str(state.root),
            "config_exists": state.config_path.exists(),
            "config_error": config_error,
            "api_key_present": api_key_present,
            "provider": provider,
            "model": model,
            "prompts_count": prompts_count,
            "prompts_file": str(state.prompts_path()),
            "docker": _docker_status(),
            "providers": PROVIDERS,
        }

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        try:
            return {"config": state.read_config_data()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/config")
    def put_config(update: ConfigUpdate) -> dict[str, Any]:
        try:
            merged = state.write_config_data(update.config)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"config": merged}

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    @app.get("/api/prompts")
    def get_prompts() -> dict[str, Any]:
        try:
            return {"prompts": state.read_prompts(), "path": str(state.prompts_path())}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/prompts")
    def put_prompts(update: PromptsUpdate) -> dict[str, Any]:
        try:
            path = state.write_prompts(update.prompts)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"prompts": state.read_prompts(), "path": str(path)}

    @app.post("/api/prompts/import")
    def import_prompts(payload: PromptsImport) -> dict[str, Any]:
        try:
            prompts = state.import_prompts_text(payload.text, replace=payload.replace)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"prompts": prompts, "path": str(state.prompts_path())}

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    @app.post("/api/generate")
    def start_generation(payload: GenerateRequest) -> dict[str, Any]:
        try:
            cfg = state.load_config()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config: {exc}")
        try:
            job = generation.start(cfg, resume=payload.resume)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return job.to_dict()

    @app.get("/api/generate")
    def get_generation() -> dict[str, Any]:
        job = generation.current()
        return {"job": job.to_dict() if job else None}

    @app.post("/api/generate/stop")
    def stop_generation() -> dict[str, Any]:
        job = generation.current()
        if job is None:
            raise HTTPException(status_code=404, detail="No generation run")
        job.stop()
        return {"job": job.to_dict()}

    @app.get("/api/generate/events")
    def generation_events(after: int = 0) -> StreamingResponse:
        job = generation.current()
        if job is None:
            raise HTTPException(status_code=404, detail="No generation run")
        return _sse(job.events, after)

    # ------------------------------------------------------------------
    # Interactive sessions
    # ------------------------------------------------------------------

    @app.post("/api/sessions")
    def create_session(payload: SessionCreate) -> dict[str, Any]:
        try:
            cfg = state.load_config()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config: {exc}")
        cfg = cfg.model_copy(deep=True)
        if payload.provider:
            cfg.agent.provider = payload.provider
        if payload.model:
            cfg.model.model = payload.model
        try:
            validate_chat_api_compatibility(cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if cfg.get_agent_provider() != "chat" and not _docker_status()["available"]:
            raise HTTPException(
                status_code=409,
                detail="Docker is not available. Start Docker, or use the chat provider.",
            )
        session = sessions.create(
            cfg,
            github_repo=(payload.github_repo or "").strip() or None,
            system=(payload.system or "").strip() or None,
        )
        return session.to_dict()

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": sessions.list()}

    def _session(session_id: str):
        try:
            return sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session = _session(session_id)
        return {**session.to_dict(), "events": session.events.snapshot()}

    @app.post("/api/sessions/{session_id}/message")
    def send_message(session_id: str, payload: SessionMessage) -> dict[str, Any]:
        session = _session(session_id)
        try:
            session.send_async(payload.text)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return session.to_dict()

    @app.post("/api/sessions/{session_id}/save")
    def save_session(session_id: str) -> dict[str, Any]:
        session = _session(session_id)
        try:
            trace_path = session.save()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {**session.to_dict(), "trace": trace_path.name}

    @app.post("/api/sessions/{session_id}/discard")
    def discard_session(session_id: str) -> dict[str, Any]:
        session = _session(session_id)
        session.discard()
        sessions.remove(session_id)
        return {"ok": True}

    @app.get("/api/sessions/{session_id}/events")
    def session_events(session_id: str, after: int = 0) -> StreamingResponse:
        session = _session(session_id)
        return _sse(session.events, after)

    @app.websocket("/api/sessions/{session_id}/term")
    async def session_terminal(websocket: WebSocket, session_id: str, cols: int = 120, rows: int = 32) -> None:
        try:
            session = sessions.get(session_id)
        except KeyError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()

        def on_output(text: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, text)

        await _wait_for_terminal_session_ready(websocket, session)
        if session.status == "error":
            await websocket.send_json({"type": "exit", "detail": "Session failed to start — see the session log."})
            await websocket.close()
            return
        try:
            session.start_terminal(cols=max(20, min(cols, 500)), rows=max(5, min(rows, 200)))
        except RuntimeError as exc:
            await websocket.send_json({"type": "exit", "detail": str(exc)})
            await websocket.close()
            return
        scrollback = session.terminal.attach(on_output)
        try:
            if scrollback:
                await websocket.send_json({"type": "stdout", "data": scrollback})

            async def pump_output() -> None:
                while True:
                    text = await queue.get()
                    await websocket.send_json({"type": "stdout", "data": text})

            async def pump_input() -> None:
                while True:
                    raw = await websocket.receive_text()
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if message.get("type") == "stdin":
                        data = message.get("data")
                        if isinstance(data, str):
                            session.terminal.write_stdin(data)

            output_task = asyncio.create_task(pump_output())
            input_task = asyncio.create_task(pump_input())
            done, pending = await asyncio.wait(
                {output_task, input_task}, return_when=asyncio.FIRST_COMPLETED
            )
            await _settle_terminal_tasks(done, pending)
        except WebSocketDisconnect:
            pass
        finally:
            session.terminal.detach(on_output)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    @app.get("/api/traces")
    def list_traces() -> dict[str, Any]:
        return {"traces": state.list_traces()}

    @app.get("/api/traces/preview")
    def preview_trace(name: str, limit: int = 400) -> dict[str, Any]:
        try:
            path = state.trace_file(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not path.exists():
            raise HTTPException(status_code=404, detail="Trace not found")
        events: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        events.append(event)
                    if len(events) >= 5000:
                        break
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        provider = detect_trace_provider(events)
        if provider == "chat":
            display: list[dict[str, Any]] = []
            for row in events:
                display.extend(summarize_chat_row(row))
        else:
            display = summarize_trace_events(provider, events)
        return {
            "name": name,
            "provider": provider,
            "event_count": len(events),
            "display": display[:limit],
            "truncated": len(display) > limit,
        }

    # ------------------------------------------------------------------
    # Static UI
    # ------------------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app
