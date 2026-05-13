# Teich - Roadmap

## Current Status: Alpha SFT Pipeline

Teich now has a usable trace-first generation and SFT preparation flow:

- Docker-backed Codex, Pi, Claude Code, and Hermes runners with concurrent prompt execution.
- Hermes runs with built-in toolsets including delegation, and exported delegated subagent sessions stay as separate trace files linked by `parent_session_id`.
- Text-only `chat` provider for structured training rows without Docker.
- YAML configuration with provider, model, prompt-file, output, MCP, and publish settings.
- CLI `init` / `generate` commands with resume support and Rich progress reporting.
- Raw trace preservation, partial-output recovery, and deterministic Docker container cleanup.
- Structured conversion to `messages` / `tools` / `metadata` rows.
- Embedded configured tool-schema snapshots in generated dataset READMEs.
- JSONL/NDJSON prompt files as the recommended generation input format.
- Multi-turn `follow_up_prompts` for chat and Docker-backed agent generation.
- `prepare_data()` + `mask_data()` as the supported trainer-first SFT path.
- `load_traces()` for fallback/manual workflows where users own chat-template rendering, filtering, tokenization, and masking.
- Generated dataset README cards that show the current Teich SFT path.

---

## Phase 1: Testing & Hardening

### 1.1 Integration Testing

- [x] Test actual Docker image build
- [x] Validate live provider smoke tests for Codex/Pi-style native extraction plus Claude Code/Hermes external traces, including Hermes delegated child-session export
- [x] Verify session files are extracted correctly with real or provider-native sessions
- [x] Verify trace format matches HF expectations on generated trace examples
- [ ] Test MCP server configuration with real servers
- [ ] Validate LM Studio / Ollama local-provider runs through Codex OSS mode

### 1.2 Error Handling

- [x] Handle Docker not installed/running
- [ ] Handle invalid OpenAI API key or unavailable local provider
- [x] Handle network timeouts during agent execution
- [x] Handle session extraction failures
- [x] Preserve partial raw traces on runner failure/interruption
- [x] Clean up orphan-prone Docker containers on failure/interruption
- [ ] Retry logic for failed prompts

### 1.3 Output Format Validation

- [x] Validate trace JSONL structure against example traces
- [x] Ensure HF trace viewer compatibility
- [x] Generate README for trace upload directories
- [x] Embed configured tool-schema snapshots in generated READMEs

---

## Phase 2: Training Data Conversion & SFT Preparation

### 2.1 Public Training Flow

Recommended path:

```python
from teich import mask_data, prepare_data

train_dataset = prepare_data(
    ["username/chat-traces", "username/tool-traces"],
    tokenizer,
    max_length=32768,
    drop_oversized_examples=True,
    chat_template_kwargs={"enable_thinking": True},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=32768,
        packing=False,
        output_dir="outputs",
    ),
)
trainer = mask_data(trainer, tokenizer=tokenizer)
```

Fallback/manual path:

```python
from teich import load_traces

dataset = load_traces("./output")
example = dataset[0]
rendered = tokenizer.apply_chat_template(
    example["messages"],
    tools=example.get("tools") or [],
    tokenize=False,
    add_generation_prompt=False,
)
tokenized = tokenizer(rendered, truncation=True, max_length=32768)
```

### 2.2 Supported Normalized Formats

- [x] **OpenAI-style chat/message format** as the primary normalized training representation.

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "reasoning_content": "..."},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "tools": [...],
  "metadata": {...}
}
```

- [x] Structured chat JSONL rows from `agent.provider: chat`
- [ ] Anthropic Messages export adapter
- [ ] Gemini export adapter

### 2.3 Field Mapping

- [x] Extract system prompts from session init / developer messages
- [x] Map user/assistant/tool messages from Codex traces
- [x] Map user/assistant/tool messages from Pi traces
- [x] Extract `reasoning_content`
- [x] Map tool calls and tool results
- [x] Apply configured tool schemas from generated README snapshots
- [x] Handle multi-turn conversations correctly

### 2.4 SFT Safety

- [x] Preserve `text` plus Teich supervision spans before trainer tokenization.
- [x] Apply masked `labels` after `SFTTrainer` tokenization with `mask_data()`.
- [x] Support optional reasoning supervision with `train_on_reasoning`
- [x] Drop oversized rows by default
- [x] Add strict marker-render invariant checks
- [x] Audit dataset labels

---

## Phase 3: Generation UX

### 3.1 Parallel Execution

- [x] Run multiple prompts concurrently
- [x] Configurable concurrency
- [x] Progress tracking for batch jobs
- [x] Stop claiming new prompts after an individual prompt fails while preserving completed outputs
- [x] JSONL/NDJSON prompt files with prompt metadata
- [x] `follow_up_prompts` list support for `agent.provider: chat`
- [x] Container-persistent follow-up support for Codex, Pi, Claude Code, and Hermes runners

### 3.2 Session Resumption

- [x] Detect completed prompts from existing outputs
- [x] Resume interrupted runs with `--resume`
- [x] Skip already-completed prompts
- [x] Keep failed/interrupted traces under `output/partials/` and exclude them from resume scanning

### 3.3 Output Formats

- [x] Raw JSONL traces
- [x] Structured chat JSONL rows
- [x] Hugging Face dataset upload
- [x] Generated dataset README
- [x] Embedded training-ready tool schema snapshot in generated README
- [x] Clean top-level `output/` policy with partial traces quarantined under `partials/`
- [ ] Parquet output option
- [ ] Train/validation split generation

### 3.4 Quality Filtering

- [x] Drop empty conversations during formatting
- [x] Drop oversized examples during formatting
- [ ] Detect failed/error sessions before training
- [ ] Workspace artifact validation
- [ ] Configurable quality thresholds

---

## Phase 4: Extended Provider & Template Support

### 4.1 OpenRouter/OpenAI-Compatible APIs

- [x] OpenRouter/config override path in current Codex runner
- [x] Text-only OpenAI-compatible `chat` provider path
- [x] Harden OpenRouter compatibility for Claude Code non-Claude models through an in-container model-rewrite proxy
- [x] Add external coding-agent runners beyond Codex/Pi
- [ ] Harden compatibility for more non-OpenAI endpoints under real runs

### 4.2 Multi-Provider Support

- [x] Modular config boundary with `agent.provider`
- [x] Codex runner
- [x] Pi runner
- [x] Chat runner
- [x] Anthropic Claude Code runner
- [x] Hermes agent runner with built-in toolsets and separate delegated subagent trace export
- [ ] Ollama/local model runner beyond Codex OSS mode

### 4.3 Chat Template Masking Coverage

- [x] Core assistant masking path
- [x] Fast assistant-mask tokenizer path
- [x] Offset marker masking path
- [x] Gemma-like structured masking path
- [x] Fallback diff masking path
- [ ] Audit tokenizer-only masking coverage for Qwen 3.5 family chat templates
- [ ] Audit tokenizer-only masking coverage for Qwen 3.6 family chat templates
- [ ] Audit Qwen 3 hybrid chat templates, including non-2507 dense and A3B variants
- [ ] Audit Qwen 3 2507 instruct and thinking chat templates
- [ ] Audit Gemma 4 tokenizer/template differences, especially `gemma-4-E2B-it` and `gemma-4-E4B-it`
- [ ] Audit Granite 4.1 chat template masking across 3B, 8B, and 30B models

---

## Phase 5: Production Polish

### 5.1 Documentation

- [x] Public README / PyPI README updated for `prepare_data()` + `mask_data()`
- [x] Generated dataset README updated for `prepare_data()` + `mask_data()`
- [x] Generation docs updated for JSONL prompt files and chat follow-up prompts
- [x] Example training script updated for `prepare_data()` + `mask_data()`
- [ ] Full API reference
- [ ] Tutorial: Creating your first dataset
- [ ] Tutorial: Fine-tuning with generated data
- [x] Example config covers Codex, Pi, Claude Code, Hermes, chat, OpenRouter, and local OpenAI-compatible endpoints
- [ ] More task-specific example configs for common use cases

### 5.2 CLI Improvements

- [ ] `validate` command to check config
- [ ] `preview` command to see what would be generated
- [ ] `status` command to check previous runs
- [ ] Better generated-run summary exports

### 5.3 Testing

- [x] Unit tests for config, CLI, runner, converter, loader, formatter, audit, collator, and SFT preparation
- [x] Format validation tests
- [x] End-to-end non-integration test pass excluding Docker/API integration
- [x] Unit coverage for provider command/env construction and native/external trace extraction
- [ ] Integration tests with real API calls mocked at provider boundary
- [ ] Docker build tests in CI

---

## Immediate Next Steps

1. **Run small real generation smoke tests** periodically for Codex, Pi, Claude Code, Hermes, and chat with 1-2 prompts each.
2. **Run a small real `prepare_data()` + `mask_data()` smoke test** against the newly generated output and the intended tokenizer.
3. **Audit generated examples manually** for tool-call rendering, reasoning supervision, follow-up turns, and empty/error sessions.
4. **Decide quality-filter policy** for failed sessions and low-value traces.
5. **Add docs/tutorials** around creating, publishing, loading, and training on a first dataset.

---

## Open Questions

1. Which Codex CLI versions should Teich explicitly support for non-interactive runs?
2. Should LM Studio and Ollama stay routed through Codex OSS mode, or get their own non-Codex runner?
3. What quality metrics should Teich filter on before training?
4. Should Teich eventually offer optional model/template presets, or keep explicit `chat_template_kwargs` only?
5. Should Claude Code's OpenRouter surrogate model be configurable, or stay fixed to a known Claude allowlist value?
6. How should Parquet and train/validation split generation fit into the trace-first workflow?
