# Teich Documentation

Start with the README for a quick overview, then use these pages for implementation detail.

## Guides

- [Teich Studio](studio.md): configure projects, manage prompts, run batches, and save interactive sessions from a browser.
- [CLI Reference](cli.md): command usage for `init`, `generate`, `extract`, `convert`, `anonymize`, `studio`, and `pool`.
- [Generation](generation.md): create new Codex, Pi, Claude Code, Hermes, or chat datasets.
- [Preparing Data](prepare-data.md): load local files, folders, Hugging Face datasets, `datasets.Dataset` objects, and source mixes.
- [Training](training.md): use Teich with TRL / Unsloth and apply response-only labels with `mask_data()`.
- [Data Format](data-format.md): understand `messages`, `tools`, metadata, native Claude context, and structured chat rows.
- [Python API](python-api.md): public functions and validation helpers.
- [Pipeline Flow](pipeline.md): generation, preparation, and masking diagrams.

## Typical Paths

Already have data:

```text
prepare_data() -> SFTTrainer -> mask_data() -> trainer.train()
```

Need new data:

```text
teich init -> edit prompts.jsonl/config.yaml -> teich generate -> prepare_data()
```

Already have local agent sessions:

```text
teich extract claude --model fable-5 -> optional HF upload -> prepare_data()
```

Want OpenAI-style JSONL for another trainer:

```text
teich extract claude --out data -> teich convert data --out teich-training.jsonl
```

Prefer a browser:

```text
teich studio -> configure/generate/steer -> save traces -> prepare_data()
```

Need full control:

```text
load_traces() -> validate_tool_calls() -> tokenizer.apply_chat_template() -> custom trainer
```
