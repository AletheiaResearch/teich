# Harbor interop for teich — design (A′: drive + ingest)

Status: approved-in-principle · Date: 2026-06-23 · Branch: `worktree-harbor-interop` (off `origin/main`)

## Goal

Run teich's agents on **Harbor-format tasks** (Terminal-Bench, datacurve-ai/deep-swe, SWE-bench-via-Harbor) and emit teich's normalized training traces **plus the task's reward** — turning the Harbor/SWE ecosystem into SFT/RL trace+reward sources.

## What changed from the first draft (Option A → A′)

Installing `harbor` (0.15.0) and reading its source overturned the premise that justified reimplementing agents:

- Harbor **ships built-in installed agents for all four teich runners**: `codex`, `claude_code`, `pi`, `hermes` (plus aider/opencode/cursor/goose/…), via `AgentFactory._AGENT_MAP` keyed by an `AgentName` enum.
- Each built-in agent already **installs the CLI into the per-task image, runs it, and copies the agent's NATIVE session JSONL out to `logs_dir/sessions/`** (codex: `cp -R $CODEX_HOME/sessions`; claude: `sessions/projects/**.jsonl`) — then also emits an ATIF `trajectory.json`.
- The task's verifier already produces **`/logs/verifier/reward.json`**.

So reimplementing `BaseInstalledAgent` per runner (old Option A) would duplicate harbor and re-introduce install-fragility for no gain. Instead:

**A′ — drive harbor's built-in agents as a library, ingest their output.** teich keeps its crown jewel (`converter.py`) by consuming the **native session files harbor already exports**, and gets the reward from `reward.json`.

## Architecture

- **Optional dependency.** `harbor` is the **`bench` extra** (`pip install teich[bench]`), gated to **Python ≥ 3.12** (`harbor>=0.15.0 ; python_full_version >= '3.12'`) since harbor requires 3.12 while teich core stays ≥3.10. Bench mode without it → a clear "install teich[bench] (needs Python 3.12+)" error. harbor is imported lazily, only on the bench path.
- **Surface.** `teich generate **--mode {prompts,bench}**` (default `prompts` = today). Source for each mode lives in config: `prompts_file`/`prompts` vs a **`bench:` block** (`bench.source` = a dir of Harbor tasks, a git repo, or an HF dataset of tasks). `--mode bench` with no `bench.source` → error.
- **Driver — `src/teich/bench/`** (imports harbor lazily):
  1. Resolve `bench.source` → task dir(s).
  2. For each task, run harbor **via its Python API** (built-in agent chosen from `agent.provider` → `AgentName`; model from `model.model`; `api` key/`base_url` passed as the agent's credentials; **docker** backend). Harbor builds the task image, runs the agent in it, runs the separate verifier.
  3. **Ingest:** read the trial's `logs_dir/sessions/*.jsonl` (native session) → existing **`converter.py`** → teich training rows; read `reward.json` → attach `reward`/`passed` (+ partial fractions); write to `output/` with the verification sidecar shape from the verifiable-tasks work.
- **Auth.** The in-container agent uses teich's existing `api` config (API key + optional `base_url`, incl. OpenRouter). **Codex ChatGPT-subscription/broker is excluded** (per-task-container conflicts).
- Existing single-image runner path (`--mode prompts`) is **untouched**.

## Vertical slice (deliverable)

1. **Confirm harbor's programmatic run API** — how to run a single task/trial in-process (the trial runner / a `run`-equivalent), select a built-in agent + model, pick the docker backend, and get the resulting `logs_dir` (with `sessions/` + `reward.json`). (We use harbor as a package, not its CLI.)
2. **Driver for codex** — `--mode bench` + `bench.source`; run harbor's `codex` agent on one real deep-swe task with `api` key auth.
3. **Ingest** — `logs_dir/sessions/*.jsonl` → `converter.py` → a teich row; `reward.json` → reward; write to `output/` (+ sidecar).
4. **One real-Docker end-to-end run** as acceptance.

## Testability boundary

- **Unit-testable (no Docker/harbor/auth):** config (`--mode`, `bench.source`), the ingest mapping (native `logs_dir/sessions` layout → `converter.py` → row), `reward.json` parsing + attachment, the "harbor not installed / Python <3.12" guard.
- **Needs Docker + harbor (no model auth):** harbor builds the task image + runs the agent/verifier plumbing.
- **Needs model auth (your run):** the agent actually solving a task end-to-end (API key in env). I prove plumbing + ingest on a captured/seeded `logs_dir`; the real solve run is yours.

## Risks / open questions

- **harbor internal-API stability** — we depend on harbor's Python API + `logs_dir`/`reward.json` layout, which can shift across versions. Pin a tested range and keep the ingest tolerant; step 1 confirms the exact API for 0.15.0.
- **Python 3.12 floor for bench** — acceptable (optional extra); core teich unaffected.
- **harbor's heavy deps** (supabase/tiktoken/…) — fine because optional.
- **Backend** — docker first; modal/daytona later via harbor config.
- **pi/hermes/claude** — should be near-free once codex ingest works (same `logs_dir/sessions` + converter path), but each agent's native layout is verified before claiming support.

## Relationship to PR #1

Independent. PR #1 (teich-native seed bundles + its own F2P/P2P verifier) stays; Harbor mode delegates env+verifier to harbor. They share `converter.py` and the reward-sidecar idea; neither depends on the other.
