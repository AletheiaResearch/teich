# Unify benchmarks under bench mode: pluggable backends (harbor + swe-bench)

## Context

teich has two overlapping notions of "run an agent on a graded task":

- **bench mode** (`generate --mode bench`) drives the external **harbor** framework over
  Harbor-format tasks and harvests native traces + a reward.
- **prompts mode** carries a *hand-rolled* SWE-bench inside `runner.py`
  (`VerificationResult`, `_materialize_seed`, `_run_verifier`, `_restore_verifier_files`,
  `_run_seed_baseline`, `_apply_f2p_p2p_reward`, `_verify_and_record`, `_route_destination`
  + `PromptInput.{seed_repo,github_repo,base_commit,verifier,verifier_files,fail_to_pass,pass_to_pass}`
  + `TasksConfig`/`SeedReference`) — ~500 lines bolted onto an already-6.3k-line file.

Both are benchmarks. This spec unifies them: **every graded benchmark is a bench backend.**
harbor stays a backend; SWE-bench becomes a real backend built on the official `swebench`
package (dataset + environment + grading) instead of the hand-rolled version, which is
deleted. Prompts mode goes back to pure trace generation.

SWE-bench is **evaluation-only**: its harness grades a *prediction*
(`{instance_id, model_patch, model_name_or_path}`) — it never runs an agent. So teich uses
swebench for the **dataset**, the **environment** (per-instance image: repo @ base_commit +
deps), and the **grading** (`eval_script` + `get_eval_report` → resolved + FAIL_TO_PASS/
PASS_TO_PASS). teich runs its **own** agent in that environment to produce the patch.

## Goals

- One `bench.sources` array; each source declares a `type` (`harbor` | `swe-bench`).
- A small backend protocol; the driver loop and the harvest (native trace → routed
  passed/failed/borderline + per-task `metadata/` + card splits, already built) are shared.
- swe-bench backend reuses the official harness; teich runs its agent in a container layered
  on swebench's instance image, rendered from a **Jinja Dockerfile template** (no inline
  jank), so the codex auth proxy + langfuse mount in cleanly.
- Per-backend optional installs: `teich[harbor]`, `teich[swe]`, `teich[harbor,swe]`.
- Honor `max_concurrency` + `timeout_seconds`; resume that survives partial runs.
- Delete the prompts-mode seed/SWE machinery. Net: less code than today.

## Non-goals (deferred)

- Migrating the **prompts-mode** monolithic runtime image to per-agent Jinja images. This
  spec only *seeds* the Jinja template (for the swe-bench agent layer); the prompts-runtime
  migration + killing its langfuse jank is a follow-up spec.
- Changing harbor's behavior (harbor still runs its own agent in its own container).

## Architecture

```
run_bench(cfg, console, resume, refresh):
  for source in cfg.bench.sources:                 # array of {type, source, ...}
      backend = get_backend(source.type)           # harbor | swe-bench (lazy-import its extra)
      backend.require()                            # clear error if teich[<type>] not installed
      tasks = backend.tasks(cfg, source)           # resolve dataset -> tasks/instances
      run tasks with bounded concurrency:
          if resume and trace exists (any split): skip
          run = backend.run(cfg, task)             # BACKEND-SPECIFIC: agent in env -> trace + rewards
          harvest(cfg, source, task, run)          # SHARED: route + write trace + metadata
```

### Backend protocol (`teich/bench/backends/base.py`)

```python
@dataclass
class BenchTask:
    id: str                  # globally unique within the source (harbor task name / swe instance_id)
    raw: Any                 # backend payload (harbor task dir / swe instance dict)

@dataclass
class BenchRun:
    native_lines: list[str]  # the agent's native trace (jsonl lines), written verbatim
    rewards: dict[str, float] | None
    metadata: dict[str, Any] # backend extras merged into the per-task metadata sidecar

class BenchBackend(Protocol):
    type: str
    def require(self) -> None: ...                       # assert the optional extra is importable
    def tasks(self, cfg: Config, source: BenchSource) -> Iterable[BenchTask]: ...
    def run(self, cfg: Config, source: BenchSource, task: BenchTask) -> BenchRun: ...
```

`get_backend(type)` is a small registry (`{"harbor": HarborBackend, "swe-bench": SweBenchBackend}`).
The harvest already exists (route by primary score → `passed/failed/borderline`, per-task
`metadata/<stem>.json`, card splits); it consumes `BenchRun` unchanged.

## Config

```yaml
bench:
  sources:
    - { type: harbor,    source: terminal-bench@2.0 }
    - { type: swe-bench, source: SWE-bench/SWE-bench_Verified, split: test }
  backend: docker            # harbor environment backend (harbor sources only)
```

- `BenchConfig` = `{sources: list[BenchSource]}` — **no back-compat**; the old single
  `source`/`repo`/`version`/`backend` fields are removed entirely.
- `BenchSource = {type, source, repo?, version?, split?, instances?, backend?}`.
  - `type`: `harbor` | `swe-bench`.
  - `source`: harbor → registry spec / local dir; swe-bench → HF dataset id (or local json/jsonl path).
  - `split` / `instances`: swe-bench dataset split and an optional instance-id allowlist.
  - `backend`: harbor EnvironmentType (default `docker`); per-source, harbor only.
- `_existing_dataset_modes` guard and per-source namespacing (below) unchanged in spirit.

## Backends

### harbor (`teich[harbor]`)

Today's driver moved behind the protocol with no behavior change: `tasks()` =
`_resolve_bench_source` + `_resolve_task_dirs`; `run()` = build TrialConfig + `Trial.create/run`,
return `BenchRun(native_lines=<pi stream / session dir>, rewards=<verifier_result rewards>,
metadata={harbor exception, ...})`. Existing harbor harvest/normalization moves into this backend.

### swe-bench (`teich[swe]`)

Two phases per instance, reusing the official `swebench` package:

1. **Environment** — `spec = make_test_spec(instance)`; `build_instance_images(...)` (base → env →
   instance, all cached by swebench) yields `spec.instance_image_key`: repo @ base_commit + the
   correct conda env/deps. Nothing is patched yet (clean repo = the task).
2. **Agent run (teich-owned)** — render a Jinja Dockerfile `FROM {{ instance_image_key }}` that
   adds the agent CLI + optional langfuse/auth blocks (see template), build it (cached per
   agent+features), `docker run` it with the codex auth proxy + langfuse env mounted; the agent
   solves `instance.problem_statement` in `/testbed`. `git -C /testbed diff` = the patch
   (swebench's setup commits an empty baseline so the diff is exactly the agent's changes).
   The agent's native session is the trace.
3. **Grade (swebench-owned)** — pass `{instance_id, model_patch: patch, model_name_or_path: <model>}`
   to swebench's `run_instance`, which builds a *clean* container from the instance image, applies
   the patch + `test_patch`, runs `eval_script`, and returns a report with `resolved` +
   FAIL_TO_PASS/PASS_TO_PASS results.

`BenchRun.rewards = {"resolved": 1.0|0.0, "fail_to_pass": <frac>, "pass_to_pass": <frac>}`;
primary score = `resolved` → routes to passed/failed (SWE-bench is binary; borderline is for
fractional-scoring backends like harbor). `metadata` carries the full report + instance_id +
the model patch reference.

### Jinja Dockerfile template (`teich/bench/templates/agent.dockerfile.j2`)

```dockerfile
FROM {{ base_image }}
{% block agent %}{{ agent_install }}{% endblock %}   # pi/codex/claude-code/hermes install
{% if langfuse %}{% include "langfuse.dockerfile.j2" %}{% endif %}
{% if auth_proxy %}{% include "auth_proxy.dockerfile.j2" %}{% endif %}
```

Rendered for teich-owned containers (here: the swe-bench agent layer; later: prompts runtime).
Optional features are includes, not inline conditionals — this is the no-jank shape the
maintainer endorsed and the seed of per-agent images. harbor is unaffected (harbor owns its image).

## Concurrency & resume

- **Concurrency:** run tasks through a bounded pool of size `cfg.max_concurrency` (default 1).
  Each task gets its own work dir + container; the resume-skip check happens at dispatch.
  `cfg.timeout_seconds` bounds each task's agent run. harbor's async `Trial.run` and the
  swe-bench docker steps both run inside a worker; backends must be safe to call concurrently
  (no shared mutable state; per-task paths). Output writes are atomic (write temp + rename).
- **Resume:** a task is skipped iff its harvested trace already exists in a split. To avoid
  collisions across sources, the trace/metadata stem is namespaced by source:
  `<source_id>-<task_id>` (e.g. `swe-bench_Verified-astropy__astropy-12907`). An unfinished
  task writes nothing and is retried — same semantics as prompts mode.

## Removal (the simplification)

Delete from `runner.py`: `VerificationResult`, `_materialize_seed`, `_fetch_seed_bundle`,
`_clone_seed_bundle`, `_clone_github_repo`, `_checkout_base_commit`, `_seed_checkout_name`,
`_github_repo_checkout_name`, `_run_verifier`, `_restore_verifier_files`, `_verifier_restore_files`,
`_build_verifier_command`, `_run_seed_baseline`, `_apply_f2p_p2p_reward`, `_was_failing`,
`_verify_and_record`, `_route_destination`, `_verification_sidecar_path`. Delete from `config.py`:
`TasksConfig`, `SeedReference`, `Config.resolve_seed_reference`, `Config.tasks`, and
`PromptInput.{seed_repo,github_repo,base_commit,verifier,verifier_files,fail_to_pass,pass_to_pass}`
+ their validators. **Prompts mode reverts to its original pre-seed behavior**: run the agent
on each prompt and write the trace — no verifier, no passed/failed routing, no reward surfacing.
After removal, `verification.py` and the converter's verification-sidecar reward-surfacing
(`_verification_reward_for_trace` + the `apply_reward_to_row`/`reward_from_sidecar_data` calls in
`convert_traces_to_training_data`) are used *only* by the deleted seed feature — remove both in
phase 3 (bench writes its own `metadata/`; it does not use `verification.py`).
`converter.NON_DATA_TRACE_DIR_NAMES` is unaffected (it lives in converter.py and scanners still use it).

## Optional dependencies

`pyproject.toml`: `harbor = ["harbor>=0.15.0 ; python_full_version>='3.12'"]`,
`swe = ["swebench>=4.1"]` (swebench requires-python is >=3.10, so — unlike harbor — the swe
extra needs no Python-version marker), and `bench` kept as an alias for `harbor` (back-compat)
or dropped. Each backend's `require()` raises a clear "install teich[harbor]" / "teich[swe]" error.

## Error handling

- Missing extra → `require()` raises `RuntimeError` with the install hint (CLI → Exit 1).
- A single task's failure (docker build, harbor/swe error, agent crash, grading error) is logged
  and skipped (the batch continues); config-level errors (bad type, unreadable source) abort.
- swe-bench grading failure → the task is recorded as `failed` with the error in metadata.

## Testing

- **Unit (no docker/network):** config parsing of `bench.sources` (+ back-compat single source);
  the backend registry + `require()` errors; swe-bench reward→split mapping; the per-source
  namespacing/resume key; Jinja template renders for each agent + feature combo (string assert,
  no build); concurrency pool honors `max_concurrency` (monkeypatched backend `run`).
- **Backend-mocked:** `run_bench` over a `sources` array with a fake backend → routed traces +
  metadata + card splits; resume skips harvested tasks; one task's failure doesn't abort others.
- **Integration (opt-in, skipped by default):** a real swe-bench `make_test_spec` + grade of a
  tiny instance with a known-good patch; a real harbor `hello-world` (existing).

## Suggested implementation phases (for writing-plans)

1. Backend protocol + registry; refactor harbor behind it; `bench.sources` array (+ back-compat);
   driver loop. No behavior change for harbor.
2. Concurrency (bounded pool) + per-source resume namespacing.
3. Delete the prompts-mode seed/SWE machinery + config fields.
4. swe-bench backend: dataset load, env build, Jinja agent layer, agent run, grading, rewards.
5. Extras split (`teich[harbor]`/`teich[swe]`) + docs/config template + card already handles splits.
