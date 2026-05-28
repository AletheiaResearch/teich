from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset, Features, Json, List, Value

from teich import prepare_data
from teich.prepare import _mix_prepared_datasets


class TinyChatTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._reverse_vocab: dict[int, str] = {}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        tool_prefix = ""
        if tools:
            tool_names = ",".join(tool["function"]["name"] for tool in tools)
            tool_prefix = f"<tools>{tool_names}</tools>"
        rendered = tool_prefix + "".join(
            f"<{message['role']}>{message.get('content', '')}</{message['role']}>" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self(rendered)
        return rendered

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True):
        input_ids: list[int] = []
        for character in text:
            token_id = self._vocab.setdefault(character, len(self._vocab) + 1)
            self._reverse_vocab[token_id] = character
            input_ids.append(token_id)
        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(input_ids)
        return output

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._reverse_vocab[token_id] for token_id in token_ids)


class ThinkingModeTokenizer(TinyChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **kwargs,
        )
        rendered = (
            f"<template enable_thinking={kwargs.get('enable_thinking')} "
            f"preserve_thinking={kwargs.get('preserve_thinking')}>"
            f"{rendered}"
        )
        if tokenize:
            return self(rendered)
        return rendered


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            }
        ]
    )


def _dataset_with_answers(prefix: str, count: int) -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"{prefix} prompt {index}"},
                    {"role": "assistant", "content": f"{prefix} answer {index}"},
                ],
                "tools": [],
            }
            for index in range(count)
        ]
    )


def _write_structured_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "prompt": "hello",
                "response": "world",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_readme_tool_snapshot(path: Path, tool_name: str) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"{tool_name} tool",
                "parameters": {
                    "type": "object",
                    "properties": {f"{tool_name}_arg": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        }
    ]
    path.write_text(
        "<details>\n"
        "<summary>Training-ready tool schema snapshot</summary>\n\n"
        "```json\n"
        f"{json.dumps(tools, indent=2)}\n"
        "```\n"
        "</details>\n",
        encoding="utf-8",
    )


def _write_source_with_tool_snapshot(root: Path, *, tool_name: str, prompt: str, response: str) -> None:
    root.mkdir(parents=True)
    (root / "trace.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "prompt": prompt,
                "response": response,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme_tool_snapshot(root / "README.md", tool_name)


def test_prepare_data_loads_local_source(tmp_path: Path):
    dataset_file = tmp_path / "chat.jsonl"
    _write_structured_dataset(dataset_file)
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        dataset_file,
        tokenizer,
        split=None,
        verbose=False,
    )

    assert prepared.num_rows == 1
    assert set(prepared.column_names) == {"text", "teich_supervised_spans"}


def test_prepare_data_loads_native_agent_traces_end_to_end(tmp_path: Path):
    prompt_wrapper = '<file name="/workspace/.teich-prompt.txt">\nBuild from file\n</file>'
    (tmp_path / "codex.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "codex-1", "source": "codex"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt_wrapper}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Codex done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "claude.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": "claude-1",
                        "message": {"role": "user", "content": "Build Claude"},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": "claude-1",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Claude done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "hermes.jsonl").write_text(
        json.dumps(
            {
                "id": "hermes-1",
                "source": "cli",
                "model": "Opus-Agent",
                "messages": [
                    {"role": "user", "content": "Build Hermes"},
                    {"role": "assistant", "content": "Hermes done"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "pi.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "pi-1"}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Build Pi"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Pi done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = prepare_data(tmp_path, TinyChatTokenizer(), split=None, verbose=False)

    assert prepared.num_rows == 4
    rendered = "\n".join(prepared["text"])
    assert "Build from file" in rendered
    assert ".teich-prompt.txt" not in rendered
    assert "Codex done" in rendered
    assert "Claude done" in rendered
    assert "Hermes done" in rendered
    assert "Pi done" in rendered


def test_prepare_data_forwards_hf_token_alias_to_loader():
    tokenizer = TinyChatTokenizer()

    with patch("teich.prepare.load_traces", return_value=_dataset()) as mock_load_traces:
        prepared = prepare_data("armand0e/ag-datagen-v2-test", tokenizer, hf_token="hf-test", verbose=False)

    assert prepared.num_rows == 1
    mock_load_traces.assert_called_once()
    assert mock_load_traces.call_args.kwargs["token"] == "hf-test"


def test_prepare_data_rejects_conflicting_token_aliases():
    tokenizer = TinyChatTokenizer()

    with pytest.raises(ValueError, match="token or hf_token"):
        prepare_data(
            "armand0e/ag-datagen-v2-test",
            tokenizer,
            token="hf-one",
            hf_token="hf-two",
            verbose=False,
        )


def test_prepare_data_accepts_source_mix_with_percentages_and_caps():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        {
            "max_examples": 10,
            "agent": {"source": _dataset_with_answers("agent", 20), "percentage": 70},
            "chat": {"source": _dataset_with_answers("chat", 20), "percentage": 30, "max_examples": 4},
        },
        tokenizer,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 10
    assert sum("agent answer" in text for text in texts) == 7
    assert sum("chat answer" in text for text in texts) == 3


def test_prepare_data_source_mix_supports_dataset_level_chat_template_kwargs():
    tokenizer = ThinkingModeTokenizer()

    prepared = prepare_data(
        {
            "reasoning": {
                "source": _dataset_with_answers("reasoning", 1),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "instruct": {
                "source": _dataset_with_answers("instruct", 1),
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    reasoning_text = next(text for text in texts if "reasoning answer" in text)
    instruct_text = next(text for text in texts if "instruct answer" in text)
    assert "<template enable_thinking=True preserve_thinking=True>" in reasoning_text
    assert "<template enable_thinking=False preserve_thinking=True>" in instruct_text


def test_prepare_data_source_mix_rejects_invalid_dataset_level_chat_template_kwargs():
    tokenizer = TinyChatTokenizer()

    with pytest.raises(TypeError, match="chat_template_kwargs must be a mapping"):
        prepare_data(
            {
                "bad": {
                    "source": _dataset(),
                    "chat_template_kwargs": ["enable_thinking", False],
                }
            },
            tokenizer,
            verbose=False,
        )


def test_prepare_data_source_mix_applies_each_source_tools_snapshot_independently(tmp_path: Path):
    tokenizer = TinyChatTokenizer()
    alpha_source = tmp_path / "alpha"
    beta_source = tmp_path / "beta"
    _write_source_with_tool_snapshot(
        alpha_source,
        tool_name="alpha_tool",
        prompt="alpha prompt",
        response="alpha answer",
    )
    _write_source_with_tool_snapshot(
        beta_source,
        tool_name="beta_tool",
        prompt="beta prompt",
        response="beta answer",
    )

    prepared = prepare_data(
        {
            "max_examples": 2,
            "alpha": {"source": alpha_source},
            "beta": {"source": beta_source},
        },
        tokenizer,
        split=None,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 2
    alpha_text = next(text for text in texts if "alpha answer" in text)
    beta_text = next(text for text in texts if "beta answer" in text)
    assert "<tools>alpha_tool</tools>" in alpha_text
    assert "beta_tool" not in alpha_text
    assert "<tools>beta_tool</tools>" in beta_text
    assert "alpha_tool" not in beta_text


def test_prepare_data_source_mix_percentages_scale_down_to_limiting_source():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        {
            "max_examples": 10,
            "agent": {"source": _dataset_with_answers("agent", 2), "percentage": 70},
            "chat": {"source": _dataset_with_answers("chat", 20), "percentage": 30},
        },
        tokenizer,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 3
    assert sum("agent answer" in text for text in texts) == 2
    assert sum("chat answer" in text for text in texts) == 1


def test_prepare_data_source_mix_percentages_keep_large_limited_ratio():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        {
            "max_examples": 1000,
            "agent": {"source": _dataset_with_answers("agent", 608), "percentage": 80},
            "chat": {"source": _dataset_with_answers("chat", 1000), "percentage": 20},
        },
        tokenizer,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 760
    assert sum("agent answer" in text for text in texts) == 608
    assert sum("chat answer" in text for text in texts) == 152


def test_prepare_data_source_mix_without_percentages_includes_all_rows_before_global_cap():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        [
            {"source": _dataset_with_answers("small", 2)},
            {"source": _dataset_with_answers("large", 10)},
        ],
        tokenizer,
        max_examples=20,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 12
    assert sum("small answer" in text for text in texts) == 2
    assert sum("large answer" in text for text in texts) == 10


def test_prepare_data_source_mix_without_percentages_trims_after_global_shuffle():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        [
            {"source": _dataset_with_answers("small", 2)},
            {"source": _dataset_with_answers("large", 10)},
        ],
        tokenizer,
        max_examples=8,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 8
    assert 0 <= sum("small answer" in text for text in texts) <= 2
    assert sum("small answer" in text for text in texts) + sum("large answer" in text for text in texts) == 8


def test_prepare_data_source_mix_normalizes_prepared_span_features():
    json_features = Features(
        {
            "text": Value("string"),
            "teich_supervised_spans": List(Json()),
            "input_ids": List(Value("int32")),
            "attention_mask": List(Value("int8")),
        }
    )
    json_span_dataset = Dataset.from_list(
        [
            {
                "text": "<assistant>agent answer</assistant>",
                "teich_supervised_spans": [{"start": 11, "end": 23, "kind": "final", "role": "assistant"}],
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
            }
        ],
        features=json_features,
    )
    structured_span_dataset = Dataset.from_list(
        [
            {
                "text": "<assistant>chat answer</assistant>",
                "teich_supervised_spans": [{"start": 11, "end": 22, "kind": "final", "role": "assistant"}],
                "input_ids": [4, 5, 6],
                "attention_mask": [1, 1, 1],
            }
        ]
    )

    mixed = _mix_prepared_datasets(
        [json_span_dataset, structured_span_dataset],
        probabilities=[0.5, 0.5],
        max_examples=2,
    )

    assert mixed.num_rows == 2
    assert mixed.features["teich_supervised_spans"] == List(Json())
    assert {mixed[index]["text"] for index in range(mixed.num_rows)} == {
        "<assistant>agent answer</assistant>",
        "<assistant>chat answer</assistant>",
    }


def test_prepare_data_plain_source_list_applies_max_examples_globally():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        [
            _dataset_with_answers("first", 5),
            _dataset_with_answers("second", 5),
        ],
        tokenizer,
        max_examples=3,
        verbose=False,
    )

    assert prepared.num_rows == 3


def test_prepare_data_plain_source_list_supports_documented_training_options():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        [
            _dataset_with_answers("first", 2),
            _dataset_with_answers("second", 2),
        ],
        tokenizer,
        max_examples=4,
        max_length=100,
        drop_oversized_examples=True,
        tokenize=True,
        chat_template_kwargs={"enable_thinking": True},
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 4
    assert set(prepared.column_names) == {"text", "teich_supervised_spans", "input_ids", "attention_mask"}
    assert any("first answer" in text for text in texts)
    assert any("second answer" in text for text in texts)
