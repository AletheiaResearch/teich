# Seed-workspace verifiable bug-fix tasks — design

Status: draft for review · Date: 2026-06-21 · Branch: `worktree-seed-workspace-verifiable-tasks`

## Goal

Let teich generate **verifiable bug-fix traces**: seed an agent's workspace from a repository that already contains a planted bug (with real git history), run the agent to fix it, then run a verifier and stamp the trace with a pass/fail reward. The trace is SFT data; the pass/fail is the RL reward.

This implements steps 3 + 5 of the pipeline the user is targeting (an LLM plants a subtle bug + writes a test that passes only when fixed → an agent fixes it, recording I/O → the test result is the reward). Step 2 (generating the bugs/tests) is **out of scope** here.

## Approved decisions (from brainstorming)

1. **teich's boundary = consume + verify.** teich takes ready-made tasks `{seed repo + prompt + verifier}` and runs them; it does not generate the bugs/tests.
2. **Verifier lives in teich.** teich runs the verifier after the agent finishes and records a binary reward in the trace metadata.
3. **Seed storage = git bundles in an HF dataset**, fetched per-task via `huggingface_hub` (with `github_repo` still available as an alternative source).

## Scope

**v1 (this spec):**
- New task fields `seed_repo` and `verifier` on prompt rows + `PromptInput`.
- Seed materialization: fetch a `git bundle` (from HF dataset, local path, or `hf://` URI) and `git clone` it into the workspace with history intact.
- Verifier execution: run a command over the post-edit workspace in the runtime container, parse a binary outcome, with a timeout.
- Anti-tamper: restore task-owned verifier files before running the verifier so the agent can't edit the oracle.
- Reward recording: a `.verification.json` sidecar next to each trace + an inline `custom` event in the trace.
- Works for all Docker agent runners (codex, pi, claude-code, hermes) via shared base-class helpers. `chat` is excluded (no workspace).

**Future (noted, not built):**
- Full SWE-bench rigor: separate `FAIL_TO_PASS` / `PASS_TO_PASS` test sets, reward = all F2P fail→pass AND all P2P stay pass; verifier in a fresh clean container rather than the agent's workspace; structured per-test parsing.
- Bug/test generation (pipeline step 2), e.g. via the `chat` provider.
- Partial-credit metrics.

## Task schema

Prompt rows (JSONL/CSV/JSON) and `PromptInput` gain:

| Field | Type | Meaning |
|-------|------|---------|
| `seed_repo` | str \| null | Starting workspace with history. One of: `hf://datasets/<owner>/<name>/<path>.bundle`; a bare key resolved against `tasks.seed_dataset` (e.g. `widgets-bug-01` → `<seed_dataset>/widgets-bug-01.bundle`); or a local path to a `.bundle`. Mutually exclusive with `github_repo`. |
| `verifier` | str \| null | Shell command run in the workspace after the agent finishes. Binary reward (see below). |
| `verifier_files` | list[str] \| null | (optional) Paths restored from the seed repo's `HEAD` before running the verifier, so the agent cannot tamper with the oracle. Defaults to none in v1. |

Existing fields (`prompt`, `system`, `follow_up_prompts`, `github_repo`) are unchanged. A row with `verifier` but no `seed_repo`/`github_repo` runs the verifier against the greenfield workspace (allowed, but unusual).

## Config

New `tasks` section on `Config`:

```yaml
tasks:
  seed_dataset: owner/my-seed-repos   # HF dataset id for resolving bare seed_repo keys
  verifier_timeout_seconds: 300       # wall-clock cap for the verifier
  restore_verifier_files: true        # restore the row's verifier_files from HEAD before verifying
```

HF auth reuses the existing `get_hf_token()` (config `publish.hf_token` or `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`/`TEICH_HF_TOKEN`). Public datasets need no token.

## Architecture & data flow

All new logic lives on the shared `DockerRuntimeRunner` base so every Docker agent inherits it.

1. **Resolve + fetch (`_resolve_seed_bundle`)** — turn `seed_repo` into a local `.bundle` path:
   - `hf://datasets/<owner>/<name>/<path>` or bare key → `huggingface_hub.hf_hub_download(repo_id, filename, repo_type="dataset", token=…)` (cached; no full-dataset pull).
   - local path → used directly.
2. **Materialize (`_prepare_workspace`, extended)** — when `seed_repo` is set, `git clone <bundle> <workspace>` (gives a normal working tree + `.git` with full history). `git bundle verify` first for a clear error on a corrupt bundle. Reject when both `seed_repo` and `github_repo` are set.
3. **Run the agent** — unchanged (existing runners).
4. **Verify (`_run_verifier`, new shared helper)** — after the agent finishes, before workspace cleanup:
   - If `restore_verifier_files`, `git -C <workspace> checkout HEAD -- <verifier_files>` (and remove any agent-added files among them) so the oracle is canonical.
   - Run the verifier in the runtime image over the post-edit workspace:
     `docker run --rm -v <workspace>:/workspace -w /workspace <image> bash -lc '<verifier>'`, capturing exit code + stdout/stderr (tail), under `verifier_timeout_seconds`.
   - `passed = (exit_code == 0)`. Timeout / crash / setup failure → `passed = false` with a reason flag.
5. **Record** — write reward (see below). The trace itself is kept regardless of pass/fail (failing fixes are still valid SFT/RL data).

```
seed_repo ──fetch──> bundle ──git clone──> workspace ──agent edits──> workspace'
                                                              │
                                          restore verifier files (anti-tamper)
                                                              │
                                                    docker run <verifier>
                                                              │
                                              passed = exit_code == 0  ──> reward
```

## Reward recording

- **Sidecar:** `<trace>.verification.json` next to the trace file (mirrors the existing hermes `.metadata.json` sidecar pattern):
  ```json
  {"verifier": "pytest -q", "passed": true, "exit_code": 0,
   "duration_s": 12.4, "timed_out": false, "stdout_tail": "…", "stderr_tail": "…",
   "seed_repo": "widgets-bug-01"}
  ```
- **Inline event:** append a teich `custom` event (`type: "teich-verification"`) into the trace so the reward travels with the conversation for providers that keep native JSONL (codex/pi).
- **Conversion:** `teich convert` surfaces `reward`/`passed` on the training row so downstream RL can read it. (Conversion wiring is part of v1.)

## Error handling

| Condition | Behavior |
|-----------|----------|
| `seed_repo` + `github_repo` both set | Config/validation error before run |
| Bundle fetch fails (network/auth/404) | Fail the session with a clear, actionable error |
| Corrupt bundle (`git bundle verify` fails) | Fail the session with a clear error |
| Private dataset, no token | Clear error naming the token env vars |
| Verifier non-zero / timeout / crash | `passed=false` (+ reason); trace kept, not discarded |
| No `verifier` on a row | Skip verification; no sidecar |

## Testing (written; not executed here per the no-tests/no-docker constraint)

- Config: parse `tasks.*`, `seed_repo`, `verifier`, `verifier_files`; reject `seed_repo`+`github_repo`.
- `_resolve_seed_bundle`: `hf://`, bare-key→dataset, and local-path resolution (mock `hf_hub_download`).
- `_prepare_workspace`: clones a real local bundle into a working tree with history (uses git only — no Docker; left for the user to run).
- `_run_verifier`: builds the expected `docker run` command; maps exit 0/non-0/timeout to the reward (mock the subprocess).
- Reward sidecar + inline event shape; `convert` surfaces the reward.

I will write these tests but **not run** them (and not build/touch Docker); you run the suite when your Docker workloads are done.

## Open decisions for your review

1. **Reward model for v1** — binary from the single `verifier` command's exit code (matches your "test script returns True/False"), with `FAIL_TO_PASS`/`PASS_TO_PASS` deferred to future. OK, or do you want the F2P/P2P split now?
2. **Where the verifier runs** — v1 runs it over the agent's post-edit workspace (with verifier-file restore for anti-tamper). The stricter SWE-bench approach (apply the agent's diff into a *fresh* clean container) is heavier; defer? 
3. **Config shape / naming** — `tasks.seed_dataset` + `seed_repo`/`verifier` on the row. Good, or different names (`seed.dataset`, `test_command`, …)?
4. **Reward surfacing** — sidecar JSON + inline `custom` event + `convert` field. Enough, or do you want a specific RL-ready output format (e.g. a flat `{prompt, completion, reward}` export)?

## Citations (web research, verified)

- git bundle / clone: https://git-scm.com/docs/git-bundle , https://git-scm.com/docs/git-clone (use `--all`; includes HEAD).
- huggingface_hub subset download: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download , https://huggingface.co/docs/huggingface_hub/guides/download
- SWE-bench schema + scoring: https://github.com/princeton-nlp/SWE-bench , SWE-bench paper (ICLR 2024).
- Verifier/reward best practice (anti-tamper, binary reward, timeout): SWE-bench / SWE-Gym / DeepSWE lineage.
