# Generation

Teich can generate new datasets by running agent CLIs in Docker or by calling an OpenAI-compatible chat API directly.

Use generation when you want Teich to create source data for you. If you already have JSONL, a Hugging Face dataset, or a `datasets.Dataset`, use [Preparing Data](prepare-data.md) instead.

If you prefer configuring prompts and steering sessions in a browser, use [Teich Studio](studio.md). It writes the same project files and output artifacts as the CLI.

If you already have local agent sessions from Claude Code, Codex, Pi, or Hermes, use `teich extract` to stage them as an anonymized dataset without running a new generation batch.

## Create a Project

```bash
teich init my-project
cd my-project
```

This creates a starter `config.yaml` and `prompts.jsonl`.

Run a batch:

```bash
teich generate -c config.yaml
```

Resume an interrupted batch:

```bash
teich generate -c config.yaml --resume
```

Teich scans completed output rows and skips prompts that already converted into training examples. Failed or interrupted agent traces are moved to `failures/` and are not treated as completed data.

## Extract Local Sessions

Extract local sessions, anonymize them, generate a compact dataset README, and optionally upload the staged folder to Hugging Face:

```bash
teich extract claude --model fable-5
```

Supported harnesses:

```bash
teich extract claude
teich extract codex
teich extract cursor
teich extract pi
teich extract hermes
```

By default, Teich looks in the provider's usual home-directory store:

- Claude Code: `~/.claude/projects`
- Codex: `~/.codex/sessions`
- Pi: `~/.pi/agent/sessions` or `~/.pi/sessions`
- Hermes: `~/.hermes/state.db`
- Cursor: `Cursor/User/workspaceStorage` and `Cursor/User/globalStorage/state.vscdb`

If the store is somewhere else, point `--sessions-dir` at the folder or file to scan. You can pass it more than once:

```bash
teich extract claude --sessions-dir /path/to/.claude --out data
teich extract claude --sessions-dir /path/to/.claude/projects --out data
teich extract codex --sessions-dir /path/to/.codex --out data
teich extract codex --sessions-dir /path/to/.codex/sessions --out data
teich extract pi --sessions-dir /path/to/.pi --out data
teich extract pi --sessions-dir /path/to/.pi/agent/sessions --out data
teich extract pi --sessions-dir /path/to/.pi/sessions --out data
teich extract hermes --sessions-dir /path/to/.hermes --out data
teich extract hermes --sessions-dir /path/to/.hermes/state.db --out data
teich extract cursor --sessions-dir /path/to/Cursor/User/workspaceStorage --out data
teich extract cursor --sessions-dir /path/to/Cursor/User/globalStorage/state.vscdb --out data
```

By default, extracted datasets are written to `data/` under the current directory. JSONL traces are staged as provider-native or recovered session files, and the generated Hugging Face dataset metadata matches `**/*.jsonl` so nested provider paths are included. Use `--out` or `--output` to choose a different folder:

```bash
teich extract codex --model gpt-5-codex --out codex-data
```

`--model` filters by provider model metadata, not by arbitrary prompt text. This keeps traces that actually ran with matching model identifiers such as `claude-fable-5` and excludes traces that only mention the model name in conversation text.

After extraction, Teich automatically scrubs API keys, emails, and home-directory usernames while preserving embedded media payloads for conversation context. It then prints the replacement counts and asks whether to upload to Hugging Face. If you need a raw, unchanged local export, pass `--no-anon` or `--no-anonymize`:

```bash
teich extract codex --sessions-dir /path/to/.codex --out raw-codex-data --no-anon
```

If you choose upload, Teich asks for a dataset repo id and uses `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `TEICH_HF_TOKEN`; if none are set, it prompts for `HF_TOKEN`.

Important: anonymization is a best-effort safety pass, not a guarantee. Review the staged data yourself before uploading or publishing it, and remove anything you would not want released.

To turn raw or extracted traces into standalone OpenAI-style JSONL rows that do not require Teich at training time, run:

```bash
teich convert data --out teich-training.jsonl
```

The output file is newline-delimited JSON with `prompt`, `messages`, `tools`, and `metadata` fields. Use this when another trainer already knows how to consume standalone OpenAI-style message rows. Use `prepare_data()` and `mask_data()` when you want Teich to render a specific tokenizer chat template and create exact response-only labels.

## Browser UI

Launch Studio from the project directory:

```bash
teich studio
```

Studio lets you edit config, manage prompts, run or resume batches, inspect traces, and save interactive sessions as dataset traces. See [Teich Studio](studio.md).

## Prompt Files

JSONL or NDJSON is recommended:

```jsonl
{"prompt":"Build a simple todo list app in React"}
{"github_repo":"armand0e/perplexica-mcp","prompt":"Improve the search flow and update tests"}
{"system":"Answer as a concise project manager.","prompt":"Draft a compact project plan"}
{"prompt":"Draft a compact project plan","follow_up_prompts":["Revise it for a solo developer","Add a risk checklist"]}
```

Each row can include:

- `prompt`: required initial user prompt
- `system`: optional prompt-specific system prompt
- `github_repo`: optional `owner/repo` checkout for Docker-backed agent runs
- `follow_up_prompts`: optional list of additional user turns
- `seed_repo`: optional git-bundle seed workspace with history (see [Verifiable bug-fix tasks](#verifiable-bug-fix-tasks)); mutually exclusive with `github_repo`
- `base_commit`: optional commit to check out after cloning `seed_repo` or `github_repo` (SWE-bench-style)
- `verifier`: optional shell command run after the agent; reward = its exit code (or F2P/P2P below)
- `verifier_files`: optional paths restored from the seed's `HEAD` before verifying (anti-tamper)
- `fail_to_pass` / `pass_to_pass`: optional test-id lists for SWE-bench-style per-test reward

`follow_up_prompts` works across providers. The `chat` provider sends each follow-up as a real additional user turn in one generated training row. Agent providers keep one Docker container alive for the full prompt sequence and resume or continue the same saved agent session for each follow-up so workspace edits, tool caches, and in-container installs remain available.

CSV, JSON, and plain text prompt files are supported, but JSONL is safer for long prompts, code fences, newlines, repository metadata, and follow-up turns.

## Config

Minimal `config.yaml`:

```yaml
agent:
  provider: codex  # codex, pi, claude-code, hermes, or chat

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access

prompts_file: prompts.jsonl

output:
  traces_dir: ./output
  sandbox_dir: ./sandbox
  failures_dir: ./failures
  pretty_name: "My Agent Traces"

publish:
  repo_id: username/my-dataset
  hf_token: hf_xxx
  private: false
```

Generated-run dataset tags are generated from provider and model. Extraction dataset cards use the extracted provider tag and omit model tags:

- `codex`, `pi`, `claude-code`, `hermes`, `cursor`: `agent-traces`, `format:agent-traces`, provider, model, `distillation`, `teich`
- `chat`: `conversational`, model, `distillation`, `teich`

If `publish.hf_token` is omitted, Teich also accepts `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `TEICH_HF_TOKEN`.

## Outputs

Provider outputs:

- `codex` / `pi`: normalized copies of native agent session JSONL files in `output/`, workspace snapshots in `sandbox/`, and a dataset `README.md`
- `claude-code`: native Claude Code transcript JSONL copied from `.claude/projects/...`, workspace snapshots in `sandbox/`, and a dataset `README.md`
- `hermes`: generated Hermes runs use Hermes' native session export shape; extracted Hermes `state.db` sessions are staged as one JSONL file per native single-session export row, including delegated subagent sessions linked by `parent_session_id`
- `cursor`: native `.cursor/projects/.../agent-transcripts/...` JSONL files are preserved when available, including MCP tool snapshots from the same project folder; recovered `state.vscdb` rows are staged as one Cursor-style session JSONL file per recovered session
- `chat`: text-only JSONL training rows in `output/` and a dataset `README.md`

`teich extract` writes provider-native or recovered session shapes to `data/` by default, then anonymizes the staged output in place before the upload prompt.

Uploaded Hugging Face dataset artifacts include:

- generated JSONL
- dataset `README.md`
- `tools.json` when a dataset-level tool snapshot is too large to embed safely in the dataset card

Generated dataset cards are intentionally short. They include Teich attribution, counts, a bounded sample, format notes, tool-schema information, and links to the maintained training docs instead of embedding trainer-specific code that may go stale.

To produce standalone OpenAI-style training rows without relying on Teich formatting and masking in your trainer, convert the staged dataset:

```bash
teich convert data --out teich-training.jsonl
```

Generation progress reports provider/model usage when Teich can retrieve it. For OpenRouter, Teich first queries the provider's generation stats API for native token and cost accounting, then falls back to harness-reported usage. If neither source is available, Teich prints `N/A`.

## Providers

### `codex`

Copies native Codex session JSONL from mounted `CODEX_HOME/sessions` and normalizes known Codex event-shape edge cases so reasoning summaries are visible and split assistant turns render as thinking before text or tool use.

Teich appends configured `tool_schema` metadata so tools remain available for training even if the model did not call them.

#### Using your ChatGPT subscription (host auth)

By default Codex runs on an API key. To run it on your ChatGPT subscription instead, point Teich at your host Codex login:

```yaml
agent:
  provider: codex
  codex:
    use_host_auth: true
    # host_auth_file: null          # defaults to $CODEX_HOME/auth.json or ~/.codex/auth.json
    # auth_dir: ./.teich/codex-auth # where the shared snapshot lives during a run
```

You must have logged in on the host first (`codex login`). When enabled, Teich:

1. Copies your host `auth.json` **once** into `auth_dir` (a single shared snapshot). It re-seeds from the host only when the host file is newer, so a token Codex has already refreshed in place is never clobbered by a stale host copy.
2. Bind-mounts that **one** shared `auth.json` into every Codex container at `/home/codex/.codex/auth.json`. All instances read and refresh the same file instead of each refreshing an independent copy (which would invalidate the others).
3. Passes **no** `*_API_KEY` env into the container, so Codex uses the subscription tokens even if an ambient `OPENAI_API_KEY` is set in your shell.

Important caveats (Codex's OAuth refresh tokens are single-use/rotating):

- **Your host login gets invalidated.** The first time any container refreshes the token, the server rotates it and your interactive `codex` login on the host stops working. Run `codex login` again on the host afterward to restore it. (Use a dedicated Codex login for batch runs if you don't want to disturb your daily one.)
- **`auth_dir` holds credentials.** Teich refuses to place it under `traces_dir`/`sandbox_dir`/`failures_dir` (those are uploaded) and drops a `.gitignore` (`*`) into it so the snapshot isn't committed. Like the output dirs, `auth_dir` is resolved relative to the directory you run `teich` from (not the config file's location).
- **Concurrency.** Sharing one file is the safest available option, but on long batches that cross the token-expiry boundary, many containers can try to refresh at once and hit `refresh_token_reused`. Teich warns when `max_concurrency > 1` with host-auth; prefer `max_concurrency: 1` for long runs. Short batches that finish before expiry never refresh and are unaffected.
- To re-seed from a fresh host login, delete `auth_dir` (or just its `auth.json`).

#### Fast mode

Codex "fast mode" is a service tier (not a model or reasoning level) that runs a supported model faster at a higher credit rate. Enable it with:

```yaml
model:
  model: gpt-5.5      # fast mode supports gpt-5.5 / gpt-5.4
  service_tier: fast
```

Teich writes `service_tier = "fast"` into the container's `config.toml`. Fast mode requires ChatGPT subscription auth (set `agent.codex.use_host_auth: true` above) and a supported model; with an API key Codex falls back to standard pricing. `service_tier` is a free-form passthrough, so other tiers (e.g. `flex`) also work.

#### Reasoning summaries

Codex reasoning models only return their chain-of-thought as opaque encrypted content plus human-readable **summaries**; the summaries are what teich records in traces (as `reasoning_text`). Codex's default summary setting can yield empty summaries (`summary: []`). To capture richer reasoning in your traces, set the summary detail:

```yaml
model:
  model: gpt-5.5
  reasoning_effort: xhigh     # depth of reasoning
  reasoning_summary: detailed # how much of it is summarized into the trace
```

Teich writes `model_reasoning_summary = "detailed"` into `config.toml`. Values are `auto | concise | detailed | none` (free-form passthrough); leave unset to use Codex's default. Note this controls the *summary* of the reasoning, not the raw chain-of-thought — Codex/OpenAI never return the full raw CoT in plaintext.

### Developer instructions / CoT narration (all agents)

The top-level `developer_instructions` config is injected into **every** agent run as additive system/developer guidance, via each agent's native mechanism:

| Agent | Mechanism |
|-------|-----------|
| codex | `developer_instructions` in `config.toml` |
| claude-code | `--append-system-prompt` |
| pi | `--append-system-prompt` |
| hermes | auto-loaded `AGENTS.md` in the workspace (appended, so a cloned repo's own `AGENTS.md` is preserved) |

It augments each agent's built-in base prompt rather than replacing it. A useful pattern for training data is to nudge the agent to narrate its reasoning in its visible output, which lands in the trace (and SFT rows) alongside Codex's reasoning summaries:

```yaml
developer_instructions: |
  Think out loud so your problem-solving process is visible. Before each tool
  call or edit, briefly explain what you're doing and why; after a command or
  test runs, state what you concluded before the next step.
```

This produces reasoning *narration* in the assistant messages — not the model's hidden raw chain-of-thought, which providers don't expose. (The `chat` provider is text-only distillation and uses per-prompt `system` instead.)

### `pi`

Copies native Pi session JSONL from mounted `/home/codex/pi-sessions`, then normalizes and validates tool-call structure before writing output.

Teich appends prompt-level system metadata and configured tool metadata as `custom` events. For OpenRouter, Teich forces Pi onto the chat/completions wire path because Pi's OpenRouter Responses adapter can stall before the first session event.

### `openclaw`

OpenClaw is supported as an imported raw trace format. Teich recognizes it when the first session event has `.openclaw` in its `cwd`, converts it with `metadata.trace_type = "openclaw"`, and does not apply Pi runner metadata snapshots.

OpenClaw is not currently a Teich runner.

### `claude-code`

Copies Claude Code's native transcript JSONL from `.claude/projects/...` so the output keeps Claude's own `user`, `assistant`, `system`, and `result` event format.

During conversion, Teich:

- normalizes split assistant fragments so thinking appears before the text or tool use it explains
- preserves Claude runtime context such as skill listings, MCP instruction deltas, permission context, date changes, hook context, and away summaries as masked `system` messages and `metadata.system_prompt`
- filters local slash-command artifacts such as `/model`
- keeps `/goal` as the actual user goal text
- turns queued prompts into real user turns
- emits schemas for advertised native Claude Code / Claude Desktop tools even if they were only declared through deferred-tool context

With OpenRouter non-Claude models, Teich runs a local in-container proxy: Claude Code sees a Claude surrogate model name, while the proxy rewrites outbound requests back to the configured model. Native assistant/result events keep provider-returned model and usage fields when Claude Code records them.

### `hermes`

Runs Hermes Agent with built-in toolsets:

```text
safe,terminal,file,skills,memory,session_search,delegation
```

Teich extracts Hermes `state.db` sessions into one JSONL file per native single-session export row. Each file contains one session object with embedded `messages`, matching the shape of Hermes' single-session export. Hermes' internal `system_prompt`, enabled toolsets, and configured tools remain metadata on each row. Delegated subagent sessions stay linked by `parent_session_id`.

### `chat`

Calls an OpenAI-compatible API directly and writes structured training rows instead of raw agent traces.

Example:

```yaml
agent:
  provider: chat

model:
  model: gpt-4.1-mini

api:
  provider: openai
  wire_api: responses
```

A generated line contains `messages`, `prompt`, optional `thinking`, final `response`, and `model`. With follow-ups, the same row includes alternating `user` and `assistant` messages, `follow_up_prompts`, per-turn `responses`, and final `response`.

## Verifiable bug-fix tasks

Seed an agent in a repository that already contains a planted bug (with real git history), let it fix the bug, then run a verifier and record a pass/fail reward. The trace is SFT data; the reward is the RL signal. Works for codex, pi, claude-code, and hermes (not `chat`, which has no workspace).

A task is a prompt row with a `seed_repo` and a `verifier`:

```jsonl
{"prompt":"The tests are failing. Find and fix the bug.","seed_repo":"widgets-bug-01","verifier":"pip install -e . >/dev/null 2>&1 && pytest -q","verifier_files":["tests/test_widgets.py"]}
```

- **`seed_repo`** is a **git bundle** (`git bundle create REPO.bundle --all`). It can be a bare key resolved against `tasks.seed_dataset` (`widgets-bug-01` → `<dataset>/widgets-bug-01.bundle`), an `hf://datasets/<owner>/<name>/<path>.bundle` URI, or a local `.bundle` path. Teich fetches it (via `huggingface_hub`, cached) and `git clone`s it into the workspace, so the agent gets the repo with full history (`git log`/`blame`/`bisect` work).
- **`verifier`** runs in the runtime container over the post-edit workspace. **Reward = its exit code** (`0` = pass) — model your test like "a script that exits 0 only when the bug is fixed." The runtime image ships Python/Node/uv but **not your repo's dependencies**, so install them as part of the command (e.g. `pip install -e . >/dev/null 2>&1 && pytest -q`); a bare `pytest` will exit non-zero and score every task as failed.
- **`verifier_files`** are restored from the seed's `HEAD` before the verifier runs, so the agent can't tamper with the oracle.

### SWE-bench-style tasks (base_commit + F2P/P2P)

A SWE-bench instance maps almost 1:1: `repo` → `github_repo`, `base_commit` → `base_commit`, `problem_statement` → `prompt`, and the held-out tests → `fail_to_pass` / `pass_to_pass`.

```jsonl
{"prompt":"Resolve the failing tests for the reported issue.","github_repo":"owner/repo","base_commit":"<sha>","verifier":"pip install -e . >/dev/null 2>&1 && pytest -rA","fail_to_pass":["tests/test_x.py::test_bug"],"pass_to_pass":["tests/test_x.py::test_other"]}
```

- **`base_commit`** is checked out after cloning either source (`github_repo` does a blobless full clone so the commit is present; bundles already carry full history).
- **`fail_to_pass` / `pass_to_pass`** switch the reward from exit-code to **per-test**: resolved = every `fail_to_pass` test goes **fail → pass** and every `pass_to_pass` test **stays passing**. teich parses the verifier's pytest output (`-rA` recommended); a test id missing from the output counts as not-passed.
- With `check_seed_baseline: true` (default), teich runs the verifier on a **pristine re-clone of the seed** to get the genuine before-state and flags **invalid tasks** (a `fail_to_pass` that already passes on the seed = no real bug; a `pass_to_pass` already failing = broken guard). Set it `false` for exact SWE-bench after-only scoring on trusted-label datasets (one fewer run per task).
- Non-pytest frameworks need a verifier that emits pytest-style `PASSED`/`FAILED <id>` lines (or per-repo parsers, like SWE-bench, can be added later).

Configure defaults under `tasks`:

```yaml
tasks:
  seed_dataset: owner/my-seed-repos   # resolve bare seed_repo keys
  verifier_timeout_seconds: 300
  restore_verifier_files: true
  route_by_result: true
  check_seed_baseline: true           # before/after baseline for true F2P/P2P transitions
```

Outputs (per task):

- the trace is routed into `output/passed/` or `output/failed/` (when `route_by_result`),
- a granular `output/verification/<name>.json` sidecar records `passed`, `exit_code`, `duration_s`, `timed_out`, a best-effort per-test map, and stdout/stderr tails — plus `fail_to_pass`/`pass_to_pass` per-test results, the `baseline`, `resolved`, and `valid_task` when F2P/P2P are used,
- `teich convert` adds `reward` (1.0/0.0) and `passed` to each training row.

Failed-verifier traces are kept (and uploaded) — they're valid RL data; filter by `passed` if you only want successful fixes for SFT.

**Auth:** HF downloads use `publish.hf_token` or `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`/`TEICH_HF_TOKEN`; public datasets need no token.

**Roadmap:** fresh-clean-container verifier isolation (apply the agent's diff onto a clean clone) and per-repo (non-pytest) result parsers, as SWE-bench does.

## Local Providers

OpenAI-compatible local endpoints can be configured with environment variables:

```bash
export TEICH_PROVIDER=LMstudio
export TEICH_MODEL=gemma-4
export TEICH_BASE_URL=http://localhost:1234/v1
export TEICH_API_KEY=llm

teich generate -c config.yaml
```

This is useful for LM Studio, Ollama-compatible proxies, or local gateway services.
