# Bench remote source — design

> **Superseded** by [`2026-06-24-unify-bench-backends-design.md`](./2026-06-24-unify-bench-backends-design.md). This uses the single `bench.source`/`repo`/`version` config; the shipped design uses the pluggable `bench.sources` array. Kept for design history — do not treat its config surface as current.

## Context

`teich generate --mode bench` runs Harbor-format benchmark tasks from `bench.source`.
Today `bench.runner._resolve_task_dirs` accepts **local directories only** (a dir with a
`task.toml`, or a dir of such task dirs) and explicitly errors on anything else with
"git/HF sources are not wired yet".

Harbor already ships a complete remote resolver: `harbor download` fetches a task or
dataset from the legacy registry (`name@version`, the ~80 sets at
`laude-institute/harbor`), the package/git registry (`org/name@ref`), an arbitrary git
repo (`--repo <url>`), or a Hugging Face dataset repo (`huggingface.co/datasets/...`).
The CLI logic lives in `harbor/cli/download.py::_download_dataset`, which builds a client
via `harbor.registry.client.factory.RegistryClientFactory` (or `PackageDatasetClient`)
and calls `await client.download_dataset(ref, output_dir=..., export=True)`.

This feature lets `bench.source` be a **remote spec** (registry name, `org/name@ref`, or a
git/HF repo), resolved by delegating to harbor's downloader, while keeping local sources
working exactly as before.

## Goal / non-goals

- **Goal:** `bench.source` accepts local paths **and** remote specs (registry / package /
  git / HF), resolved into local task dirs before the existing run loop. One coherent path.
- **Non-goal:** reimplementing git/HF fetching, caching policy beyond "reuse if present /
  refresh on request", or any change to harvest, reward, resume, or the
  separate-dataset-per-project model.

## Approach

Delegate to harbor's in-process Python API (chosen over shelling out to the `harbor` CLI,
which would depend on the CLI on PATH and lose structured errors, and over reimplementing
clone/fetch in teich).

## Config surface (`BenchConfig` in `config.py`)

Keep `source` as the single primary field; add two optional fields mirroring
`harbor download`:

```yaml
bench:
  source: terminal-bench@2.0   # local path | name@version | org/name@ref | (dataset name when repo set)
  repo: null                   # optional git/HF registry URL; then `source` is the dataset name within it
  version: null                # optional dataset version (or encode as name@version in source)
  backend: docker              # unchanged
```

- `source: str | None` (existing), `repo: str | None` (new), `version: str | None` (new).
- Version precedence: if `source` already contains `@<version>`, that wins and `bench.version`
  is ignored; `bench.version` only applies when `source` has no `@`.
- `config.example.yaml` / `CONFIG_TEMPLATE` documented; regenerate so the match test passes.

## Resolution logic (`bench/runner.py`)

`_resolve_task_dirs(source)` keeps its current local behavior. A new resolver runs first
in `run_bench`, before the task loop:

```python
def _resolve_bench_source(cfg) -> Path:
    src = (cfg.bench.source or "").strip()
    # 1. existing local path -> return as-is (unchanged behavior)
    if Path(src).expanduser().exists(): return Path(src).expanduser()
    # 2. otherwise it's a remote spec -> download via harbor, return the exported dir
    return _download_remote_source(cfg)        # runs inside asyncio.run
```

Detection inside `_download_remote_source`, mirroring `_download_dataset`:

| Condition | Harbor client | ref |
|---|---|---|
| `bench.repo` set | `RegistryClientFactory.create(repo=cfg.bench.repo)` | `source[@version]` (dataset name in repo) |
| `/` in `source` | `PackageDatasetClient()` | `source` or `source@version` (default `@latest`) |
| else | `RegistryClientFactory.create()` (default legacy registry) | `source[@version]` |

Then `await client.download_dataset(ref, overwrite=<refresh>, output_dir=<cache>, export=True)`.
Use the returned item paths to locate the exported dataset dir, then feed it to the
existing `_resolve_task_dirs` so the rest of the pipeline is unchanged.

`run_bench` becomes: `source_dir = _resolve_bench_source(cfg)` (replacing the direct
`_resolve_task_dirs(source)` call), then `task_dirs = _resolve_task_dirs(source_dir)`.

## Caching & lifecycle

- Export into `<bench_dir>/sources/<slug>`, where `bench_dir` defaults to a `bench/` dir
  beside `traces_dir` (a sibling of the dataset, parallel to sandbox/failures, never inside
  output) and `<slug>` is derived from the spec (e.g. `terminal-bench@2.0` → `terminal-bench-2.0`).
- If the cache dir already has resolvable tasks, **reuse it** (no re-download) so runs are
  idempotent and `--resume` doesn't re-fetch.
- Add a `--refresh` flag to `generate` (passed through to `run_bench`) that forces a
  re-download (`overwrite=True`). Default false.

## Error handling

- Unknown dataset / network failure / auth failure → a single `RuntimeError` with an
  actionable message; the CLI's existing `except RuntimeError -> Exit(1)` surfaces it cleanly.
- Note in docs that HF/private registries may require `HF_TOKEN` in the environment.
- An empty/whitespace `bench.source` keeps its current "requires bench.source" error.

## Testing

- **Unit (no network):** spec detection — local path vs `name@version` vs `org/name@ref`
  vs `repo`-set — asserting the chosen client kind + computed ref + cache slug. Mock
  `RegistryClientFactory` / `PackageDatasetClient` so `download_dataset` is a stub that
  writes a fake exported task tree; assert `run_bench` resolves and runs over it.
- **Reuse:** a populated cache dir is reused (the mocked downloader is not called again)
  unless `--refresh`.
- **Integration (opt-in, like the Docker tests):** a real `harbor download` of a tiny set
  (`hello-world@1.0`), gated/skipped by default.

## Backwards compatibility

Local `bench.source` paths behave exactly as today (the local-path branch is checked
first). New config fields default to `None`, so existing configs are unaffected.
