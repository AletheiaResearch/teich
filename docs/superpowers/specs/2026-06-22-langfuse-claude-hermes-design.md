# Langfuse tracing for Claude Code and Hermes

Extend the Codex Langfuse tracing (PR #3) to the other agents that have a Langfuse
integration: **Claude Code** and **Hermes**. Pi has no Langfuse integration and
`chat` is a direct-API path — both out of scope.

## Shared config

Generalize `CodexLangfuseConfig` into a reusable `LangfuseConfig`
(`enabled`, `public_key`, `secret_key`, `base_url`; when `enabled`, all three
credentials required and non-blank). Add `agent.langfuse: LangfuseConfig` to
`AgentConfig`. Add `AgentConfig.effective_langfuse` returning
`codex.langfuse if codex.langfuse.enabled else langfuse`, so:
- existing `agent.codex.langfuse` configs keep working (back-compat), and
- the shared `agent.langfuse` block drives all three agents.

Each runner maps the shared config to its agent's env-var names.

## Env wiring

Add `_langfuse_env_items()` to `ExternalCliRunner` (default `[]`), included in the
existing env loop in `_build_external_docker_base_command` (`runner.py:2923`).
Overrides:
- **Claude:** `TRACE_TO_LANGFUSE=true`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`
- **Hermes:** `HERMES_LANGFUSE_PUBLIC_KEY`, `HERMES_LANGFUSE_SECRET_KEY`, `HERMES_LANGFUSE_BASE_URL`

Add `--add-host host.docker.internal:host-gateway` when the langfuse `base_url`
is host-local (mirrors codex).

## Claude Code (Langfuse Stop hook)

The Langfuse Claude Code integration is a `Stop` hook that runs
`python3 ~/.claude/hooks/langfuse_hook.py` after each response, using the
`langfuse` Python SDK. The script is inline copy-paste code in the docs (no
package/repo), so teich vendors it.

- **Image:** `pip install "langfuse>=4.0,<5"` into `/opt/venv` (the container's
  `python3` resolves there).
- **Vendored:** `langfuse_hook.py` committed into teich package data.
- **Per session** (in the Claude home prep): seed
  `CLAUDE_HOME/hooks/langfuse_hook.py` and write `CLAUDE_HOME/settings.json`
  registering `Stop -> python3 ~/.claude/hooks/langfuse_hook.py`. Env vars via `-e`.

## Hermes (built-in plugin)

Hermes ships an `observability/langfuse` plugin (bundled, "not enabled" by
default). `hermes plugins enable` writes `~/.hermes/config.yaml plugins.enabled`,
but teich's `hermes chat` passes `--ignore-user-config`, which ignores that file.

- **No image change** (plugin already bundled via the hermes-agent clone).
- **Per session** (when langfuse enabled): write `HERMES_HOME/config.yaml` with
  `plugins.enabled: [observability/langfuse]`, and **omit `--ignore-user-config`**
  from the `hermes chat` command. Model/provider still come from CLI flags
  (`--provider`/`--model`), so the controlled minimal config only adds the plugin.
  Env vars via `-e`.

## Testing

- **Unit:** `LangfuseConfig` validation + `effective_langfuse` fallback; per-agent
  `_langfuse_env_items`; Claude `settings.json`/hook seeding; Hermes `config.yaml`
  write + `--ignore-user-config` drop.
- **Integration (acceptance, per agent):** enable langfuse pointing `base_url` at a
  mock Langfuse, run a real turn against a mock model, assert a trace POST.

## Must-verify in implementation (not assumed)

1. Claude Code Stop hook fires in headless `claude -p` mode (hook trust — the
   Codex lesson where non-interactive `exec` silently skipped hooks).
2. Hermes plugin loads + traces with `--ignore-user-config` dropped + minimal config.
3. Container `python3` resolves to the venv with `langfuse` installed.

## Out of scope

Pi (no integration), `chat` (direct-API), and richer trace metadata (mapping
teich `prompt_id`/`session_id` into Langfuse tags).
