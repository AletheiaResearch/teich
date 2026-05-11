# Next Audit Targets

- [x] Exercise a real or close-to-real Unsloth `SFTTrainer` flow and inspect columns immediately before `mask_data()`.
  - Added regression coverage for the destructive text-tokenization/remove-columns path and the `tokenize=True` skip-prepare path.
- [x] Test the recommended README flow against real Qwen/Gemma tokenizers with current TRL/Unsloth versions.
  - Ran a tokenizer-only smoke with `Qwen/Qwen2.5-0.5B-Instruct`; `prepare_data(tokenize=True)` preserved Teich columns and `mask_data()` produced compact labels.
  - Ran a tokenizer-only smoke with `unsloth/Qwen3.5-0.8B`; unique tool output stayed masked and assistant reasoning/tool-call/final-answer targets were supervised.
  - Ran a tokenizer-only smoke with `google/gemma-4-31B-it`; `prepare_data(tokenize=True)` preserved Teich columns, `mask_data()` produced compact labels, and unique tool output stayed masked.
  - Existing real Jinja template tests continue to cover local Gemma-style rendering.
- [x] Audit fallback masking leakage for tool responses in templates where tool outputs are embedded inside assistant/model turns.
  - Fixed generic fallback subtraction for `<tool_response>...</tool_response>` plus Gemma-style delimiters.
- [x] Audit runner/converter quality filters for failed sessions, malformed tool calls, and low-value traces before training.
  - Added loader filtering for rows without assistant content, reasoning, or tool calls before preparation.

Verification: `uv run --extra dev pytest -q` -> `182 passed, 10 skipped`.
