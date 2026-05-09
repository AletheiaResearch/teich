from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Json, List, Value
from huggingface_hub import snapshot_download

from .converter import convert_traces_to_training_data


def _trace_directory(root: Path, split: str | None) -> Path:
    if split:
        candidate = root / split
        if candidate.is_dir():
            return candidate
    return root


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    try:
        return Dataset.from_list(rows, on_mixed_types="use_json")
    except TypeError as exc:
        if "on_mixed_types" not in str(exc):
            raise
    features = Features(
        {
            "prompt": Value("string"),
            "messages": List(Json()),
            "tools": List(Json()),
            "metadata": Json(),
        }
    )
    return Dataset.from_list(rows, features=features)


def _normalize_tools_snapshot(tools: Any) -> list[dict[str, Any]]:
    if isinstance(tools, list):
        return [tool for tool in tools if isinstance(tool, dict)]
    return []


def _load_tools_snapshot_from_readme(root: Path) -> list[dict[str, Any]]:
    readme_path = root / "README.md"
    if not readme_path.is_file():
        return []
    try:
        readme = readme_path.read_text(encoding="utf-8")
    except OSError:
        return []
    summary = "<summary>Training-ready tool schema snapshot</summary>"
    summary_index = readme.find(summary)
    if summary_index < 0:
        return []
    fence_start = readme.find("```json", summary_index)
    if fence_start < 0:
        return []
    json_start = readme.find("\n", fence_start)
    if json_start < 0:
        return []
    fence_end = readme.find("```", json_start + 1)
    if fence_end < 0:
        return []
    try:
        tools = json.loads(readme[json_start:fence_end].strip())
    except json.JSONDecodeError:
        return []
    return _normalize_tools_snapshot(tools)


def _load_tools_snapshot(root: Path) -> list[dict[str, Any]]:
    tools_from_readme = _load_tools_snapshot_from_readme(root)
    if tools_from_readme:
        return tools_from_readme
    candidates = [root / "tools.json"]
    if root.is_dir():
        candidates.extend(path for path in root.rglob("tools.json") if path.is_file())
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            tools = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        normalized = _normalize_tools_snapshot(tools)
        if normalized:
            return normalized
    return []


def _apply_tools_snapshot(rows: list[dict], tools: list[dict[str, Any]]) -> list[dict]:
    if not tools:
        return rows
    updated_rows: list[dict] = []
    for row in rows:
        updated = dict(row)
        updated["tools"] = tools
        updated_rows.append(updated)
    return updated_rows


def load_traces(
    source: str | Path,
    split: str | None = "train",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    max_examples: int | None = None,
) -> Dataset:
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative.")
    source_path = Path(source)
    if source_path.exists():
        root = source_path
    else:
        root = Path(
            snapshot_download(
                repo_id=str(source),
                repo_type="dataset",
                revision=revision,
                token=token,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_dir=str(local_dir) if local_dir is not None else None,
                allow_patterns=["*.jsonl", "**/*.jsonl", "README.md", "tools.json", "**/tools.json"],
            )
        )
    traces_dir = root if root.is_file() else _trace_directory(root, split)
    rows = convert_traces_to_training_data(traces_dir)
    if not rows:
        location = traces_dir if traces_dir != root else root
        if split and traces_dir == root and root.is_dir():
            raise ValueError(f"No trace files found in {location} for split '{split}'.")
        raise ValueError(f"No JSONL trace or training data files found in {location}.")
    snapshot_root = root if root.is_dir() else root.parent
    rows = _apply_tools_snapshot(rows, _load_tools_snapshot(snapshot_root))
    dataset = _dataset_from_rows(rows)
    if max_examples is not None:
        dataset = dataset.shuffle(seed=3407)
        limit = min(max_examples, dataset.num_rows)
        dataset = dataset.select(range(limit))
    return dataset
