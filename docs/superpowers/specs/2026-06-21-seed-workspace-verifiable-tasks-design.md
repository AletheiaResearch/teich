# Seed-workspace verifiable bug-fix tasks — design

Status: approved · Date: 2026-06-21 · Branch: `worktree-seed-workspace-verifiable-tasks`

## Goal

Let teich generate **verifiable bug-fix traces**: seed an agent's workspace from a repository that already contains a planted bug (with real git history), run the agent to fix it, then run a verifier and record a reward + granular per-test results. The trace is SFT data; the reward is the RL signal.

This implements steps 3 + 5 of the target pipeline (an LLM plants a subtle bug + writes tests that pass only when fixed → an agent fixes it, recording I/O → the test result is the reward). Step 2 (generating the bugs/tests) is out of scope.

## Approved decisions

1. teich's boundary = **consume + verify** (not generate the bugs/tests).
2. Verifier lives in teich; records reward + per-test detail.
3. Seed storage = **git bundles in an HF dataset** (with `github_repo` still available as an alternative).
4. **Reward = the `verifier` command's exit code** (0 = pass) — matches "a test script that returns True when fixed", correct by inspection, framework-agnostic, no fragile parsing. **Capture the most granular data as metadata** (full stdout/stderr, exit code, duration, best-effort per-test breakdown) — but the reward never depends on the parse. True `FAIL_TO_PASS`/`PASS_TO_PASS` (which requires running the tests on the buggy seed *before* the agent, plus a tested per-framework parser) is **v2**, deliberately not shipped half-done.
5. Verifier isolation = **middle**: run in the agent's post-edit workspace but restore test/verifier files from the seed repo's `HEAD` first (anti-tamper). Fresh-clean-container isolation is a documented future upgrade.
6. Output organization = route verified traces into `output/passed/` and `output/failed/`, with a parallel `output/verification/<trace>.json` granular record.

## Scope

**v1 (this spec):**
- New task fields on prompt rows + `PromptInput`: `seed_repo`, `verifier`, `verifier_files`.
- Seed materialization: fetch a `git bundle` (HF dataset key / `hf://` URI / local path) and `git clone` it into the workspace with full history.
- Verifier: restore `verifier_files` from `HEAD`, run the command over the workspace in the runtime container, capture exit code + stdout/stderr + duration + best-effort pytest per-test breakdown (metadata only), with a timeout.
- Reward: `passed = (exit_code == 0)`. Timeout / crash / patch-apply failure → `passed = false` (+ reason). The per-test parse is recorded but never gates the reward.
- Output: route the trace into `output/passed/` or `output/failed/`; write `output/verification/<trace>.json` (full record); append an inline `teich-verification` `custom` event; surface `reward`/`passed` in `teich convert`.
- All Docker agent runners (codex, pi, claude-code, hermes) via shared `DockerRuntimeRunner` helpers. `chat` excluded (no workspace). Behavior is unchanged for rows with no `verifier` (flat output as today).

**Future / v2 (noted, not built):** real `FAIL_TO_PASS`/`PASS_TO_PASS` with a before-run on the buggy seed + a tested per-framework parser (so the transition, not just the after-state, drives a regression-aware reward); fresh-clean-container verifier isolation (apply agent diff onto a clean clone); bug/test generation (pipeline step 2); partial-credit reward.

## Task schema

| Field | Type | Meaning |
|-------|------|---------|
| `seed_repo` | str \| null | Starting workspace with history. `hf://datasets/<owner>/<name>/<path>.bundle`, a bare key resolved against `tasks.seed_dataset` (`widgets-bug-01` → `<seed_dataset>/widgets-bug-01.bundle`), or a local `.bundle` path. Mutually exclusive with `github_repo`. |
| `verifier` | str \| null | Shell command run in the workspace after the agent. Reward = its exit code (0 = pass). |
| `verifier_files` | list[str] \| null | Paths restored from seed `HEAD` before verifying (anti-tamper). Defaults to none. |

(v2 will add `fail_to_pass` / `pass_to_pass` test-id lists once before/after runs + a tested parser back a regression-aware reward; not in v1 to avoid a two-list API whose halves behave identically.)

Existing fields (`prompt`, `system`, `follow_up_prompts`, `github_repo`) unchanged.

## Config

```yaml
tasks:
  seed_dataset: owner/my-seed-repos   # HF dataset id for resolving bare seed_repo keys
  verifier_timeout_seconds: 300       # wall-clock cap for the verifier
  restore_verifier_files: true        # restore the row's verifier_files from HEAD before verifying
  route_by_result: true               # write traces into output/passed | output/failed
```

HF auth reuses `get_hf_token()` (config `publish.hf_token` or `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`/`TEICH_HF_TOKEN`); public datasets need no token.

## Architecture & data flow

New logic on `DockerRuntimeRunner` (shared by all Docker agents).

1. **`_resolve_seed_bundle(seed_repo)`** → local `.bundle` path: `hf://`/bare-key → `hf_hub_download(repo_id, filename, repo_type="dataset", token=…)` (cached, single-file); local path used as-is.
2. **`_prepare_workspace` (extended)** — when `seed_repo` set: `git bundle verify` then `git clone <bundle> <workspace>` (normal working tree + `.git` history). Reject `seed_repo` + `github_repo` together.
3. **Run the agent** — unchanged.
4. **`_run_verifier(workspace, prompt_input)` (new)** — after the agent finishes, before cleanup:
   - if `restore_verifier_files`: `git -C <workspace> checkout HEAD -- <verifier_files>` (+ remove agent-added files among them).
   - run `docker run --rm -v <workspace>:/workspace -w /workspace <image> bash -lc '<verifier>'`; capture exit/stdout/stderr/duration under `verifier_timeout_seconds`.
   - best-effort parse pytest-style per-test results → `{test_id: passed}` (metadata only; never gates reward).
   - `passed = (exit_code == 0)`. Timeout/crash → `passed=false` with a reason.
5. **Record** — `VerificationResult` → routing + sidecar + inline event + convert field.

Notes (from review): `output/verification/` joins the `{"partials","failures"}` exclusion set so it's never scanned as trace data. Failed-verifier traces **are** included in uploads (wanted for RL; SFT-only users filter by `passed`). `git clone <bundle> <dest>` requires `dest` to be empty/nonexistent — clone into a fresh subdir of the mkdtemp workspace root, consistent with the existing `github_repo` checkout handling.

```
seed_repo ─fetch→ bundle ─git clone→ workspace ─agent edits→ workspace'
                                                     │ restore verifier_files (HEAD)
                                                     ▼
                                          docker run <verifier>
                                                     │ parse per-test + exit code
                                                     ▼
                                  passed (F2P&P2P or exit==0) → reward
```

## Reward recording

- **Routing:** trace written to `output/passed/<name>.jsonl` or `output/failed/<name>.jsonl` (only when a verifier ran and `route_by_result`; otherwise flat). Resume/convert/README rglob `output/`, so both are picked up; `verification` is excluded from trace scans like `partials`/`failures`.
- **Sidecar:** `output/verification/<name>.json`:
  ```json
  {"verifier": "pytest -rA", "passed": true, "exit_code": 0, "duration_s": 12.4,
   "timed_out": false, "seed_repo": "widgets-bug-01",
   "tests": {"tests/test_x.py::test_bug": "passed"}, "stdout_tail": "…", "stderr_tail": "…"}
  ```
  (`tests` is best-effort pytest parsing — informational only; `passed` comes solely from `exit_code`.)
- **Inline:** append a `custom` event `type: "teich-verification"` into the trace.
- **Convert:** `teich convert` surfaces `reward` (1.0/0.0) and `passed` on each training row.

## Error handling

| Condition | Behavior |
|-----------|----------|
| `seed_repo` + `github_repo` both set | Validation error before run |
| Bundle fetch fails (network/auth/404) | Fail session, clear error |
| Corrupt bundle (`git bundle verify`) | Fail session, clear error |
| Private dataset, no token | Clear error naming token env vars |
| Verifier non-zero / timeout / crash | `passed=false` (+ reason); trace kept (valid RL data) |
| No `verifier` on a row | Skip verification; flat output as today |

## Testing (written; not executed here per the no-tests/no-docker constraint)

- Config: parse `tasks.*` and the new row fields; reject `seed_repo`+`github_repo`.
- `_resolve_seed_bundle`: `hf://`, bare-key→dataset, local-path (mock `hf_hub_download`).
- `_prepare_workspace`: clone a real local bundle → working tree with history (git only; no Docker).
- `_run_verifier`: builds expected `docker run`; maps exit 0/non-0/timeout to reward (mock subprocess); pytest parser unit tests (metadata only).
- Recording: routing into passed/failed, sidecar shape, inline event, convert `reward` field.

Tests are written but **not run**; no Docker is built or invoked. You run the suite when your Docker work is done.

## Citations (verified web research)

- git bundle/clone (`--all` includes HEAD): https://git-scm.com/docs/git-bundle , https://git-scm.com/docs/git-clone
- huggingface_hub single-file/subset download + cache: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download , https://huggingface.co/docs/huggingface_hub/guides/download
- SWE-bench schema + F2P/P2P scoring, verifier anti-tamper/binary-reward: https://github.com/princeton-nlp/SWE-bench (+ SWE-bench ICLR 2024 paper; SWE-Gym/DeepSWE lineage)
