# Harbor interop for teich — design (Option A)

Status: draft for review · Date: 2026-06-23 · Branch: `worktree-harbor-interop` (off `origin/main`)

## Goal

Let teich run its agents on **Harbor-format tasks** (the format used by Terminal-Bench, datacurve-ai/deep-swe, and SWE-bench-via-Harbor) and emit teich's normalized training traces **plus the task's reward**. This turns the entire Harbor/SWE ecosystem into trace + reward sources for SFT/RL, while keeping teich's native-trace normalization and its pi/hermes runners (which Pier/Harbor don't ship).

Approved approach: **Option A** — wrap each teich runner as a Harbor agent that runs **inside the task's environment image**; Harbor owns the environment + verifier/reward; teich keeps `converter.py`. Build **vertical-slice-first** (codex on one real task) before generalizing.

## Verified Harbor contract (from harbor-framework/harbor + datacurve-ai/deep-swe)

- A task is a directory: `task.toml` + `instruction.md` + `environment/` (Dockerfile/compose/image) + `tests/` (verifier `test.sh` + grader) + optional `solution/`. `task.toml` sections: `[task]`, `[metadata]` (freeform — SWE fields `repository_url`/`base_commit_hash`/`language` live here), `[environment]`, `[agent]`, `[verifier]`, `[solution]`, `artifacts`.
- **The agent runs INSIDE the per-task environment container.** The **repo** is baked into that image (`environment/Dockerfile` clones `repository_url` + checks out `base_commit_hash`); the **agent CLI is installed at runtime** by the agent's `install()` (not baked in).
- **Verifier**: `[verifier] environment_mode="separate"` → grading in a pristine container. The agent's git diff is captured (`pre_artifacts.sh` → `model.patch`), reapplied on a clean checkout, F2P/P2P run → reward written to **`/logs/verifier/reward.json`** (binary + partial fractions; `reward.txt` fallback).
- **Agent interface**: `BaseAgent` (`name`/`version`/`setup`/`run(instruction, environment, context)`) and `BaseInstalledAgent` (adds `install(environment)` + `exec_as_root`/`exec_as_agent` helpers + a `with_prompt_template` decorator). Run via `harbor run`/`harbor trials start -p <task> -a <agent> -m <model>`; backends docker/modal/daytona. `harbor` is a PyPI package.

## Architecture

**Surface / naming:** `teich generate **--mode {prompts,bench}**` (default `prompts`, = today's behavior). The *source* for each mode lives in config, consistent with prompts: `prompts_file`/`prompts` for prompts mode, a **`bench:` block** (`bench.source` = a dir of Harbor tasks, a git repo, or an HF dataset of tasks) for bench mode. `--mode bench` with no `bench.source` is an error. No path on the CLI, no separate command; "Harbor" stays an internal *format* detail. Internal package is `src/teich/bench/` with a Harbor-format adapter (`bench/harbor.py`); `agent.provider`, `model`, and `api` auth come from config as usual.

**Auth:** the in-container agent authenticates via teich's existing **`api` config** (API key + optional `base_url`, incl. OpenRouter / OpenAI-compatible). The Codex ChatGPT-subscription/broker path is **explicitly excluded** for bench mode (per-task containers + broker host-wiring conflict).

New package `src/teich/bench/`:

- **`<Runner>HarborAgent(BaseInstalledAgent)`** per teich runner (codex first):
  - `install(environment)` — `exec_as_root` for system deps (e.g. node), `exec_as_agent` to install the CLI (`npm i -g @openai/codex`) and seed auth/config (CODEX_HOME/auth.json or API-key env, `config.toml`). Optional teich extras (Langfuse plugin, proxies) are **opt-in** and skipped for air-gapped tasks.
  - `run(instruction, environment, context)` — exec the headless CLI (`codex exec --skip-git-repo-check …`) against `instruction.md` in the task workdir, teed to `/logs/agent/`.
  - **harvest** (post-run hook) — read the agent's native session JSONL from the container, feed it to existing `converter.py` → teich training rows; read `/logs/verifier/reward.json` and attach `reward`/`passed`/partials.
- **Driver** — `teich generate --bench <source>` runs Harbor with the teich agent over a task/dataset and writes teich rows + reward sidecars into `output/`. Reuses `converter.py` and the reward-sidecar shape from the verifiable-tasks work.
- teich gains a dependency on **`harbor`**; the existing single-image runner path is untouched (Harbor tasks are a separate path).

## Vertical slice (this spec's deliverable)

1. **Confirm the real Harbor agent API** — `pip install harbor`, read the actual `BaseInstalledAgent` import path + how a custom agent is registered/discovered (entry point vs `--agent` import path). Write a hello-world custom agent that runs on `examples/tasks/hello-world`.
2. **CodexHarborAgent** — install codex + seed auth, run `codex exec` against one real **deep-swe** task (e.g. `abs-module-cache-flags`), let Harbor build the task image + run its separate verifier.
3. **Harvest** — pull codex's native session out of the container → `converter.py` → a teich row; read `reward.json` and attach the reward.
4. **One end-to-end run** on real Docker as the acceptance test.

## Testability boundary (explicit)

- **Unit-testable (no Docker/auth):** task.toml/`reward.json` parsing, the harvest→converter mapping, reward attachment, CLI-invocation construction.
- **Needs Docker (no auth):** `install()` into a built task image; Harbor environment plumbing; reading the native session out of the container.
- **Needs auth (your run):** the agent *actually fixing* the task end-to-end (codex needs an API key/ChatGPT subscription). I'll prove plumbing + a mocked/seeded run; the real fix run is yours.

## Risks / open questions

- **Custom-agent registration API** — exact mechanism (entry point? import path?) unconfirmed; step 1 resolves it. If harbor can't load an external agent cleanly, fall back to driving harbor programmatically or vendoring the agent base.
- **Install fragility** — teich's heavier tooling (Langfuse, codex auth broker, proxies, `host.docker.internal`) into heterogeneous/air-gapped task images is the known cost; keep it opt-in, slice uses the minimum.
- **Auth into the task container** — decided: use the `api` key (+ `base_url`) only; the Codex subscription/broker is excluded to avoid per-task-container conflicts.
- **pi/hermes** — deferred to after the codex slice proves the contract.

## Out of scope (this slice)

pi/hermes Harbor agents; air-gapped/allowlist tooling; a full SWE-bench/deep-swe collection importer; the native teich seed-bundle verifier path (that's the separate PR #1 — Harbor tasks bring their own verifier).

## Relationship to PR #1

Independent. PR #1 (seed bundles + teich's own F2P/P2P verifier) stays teich-native; Harbor interop is a parallel path where Harbor owns the environment + verifier. They share `converter.py` and the reward-sidecar idea but don't depend on each other.
