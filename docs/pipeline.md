# Pipeline Flow

This page describes how Teich moves from raw data to trainer labels.

## Combined Flow

```mermaid
flowchart TD
    A["Raw source<br/>HF repo, local traces, Dataset, source mix, or prompts"] --> B{"Need generation?"}
    B -->|"yes"| C["teich generate"]
    B -->|"existing local sessions"| X["teich extract"]
    B -->|"want standalone JSONL"| Y["teich convert"]
    B -->|"no"| D["prepare_data"]
    C --> C1["Run provider<br/>codex, pi, claude-code, hermes, or chat"]
    C1 --> C2["Write raw/native traces or structured chat rows"]
    C2 --> C3["Write compact README + optional tools.json"]
    C3 --> D
    X --> X1["Copy local sessions<br/>claude, codex, cursor, pi, or hermes"]
    X1 --> X2["Filter by --model metadata when requested"]
    X2 --> X3["Anonymize staged data + write compact README"]
    X3 --> X4{"upload to Hugging Face?"}
    X4 -->|"yes"| X5["Upload JSONL + README/tools metadata"]
    X4 -->|"no"| D
    X5 --> D
    X3 --> Y
    Y --> Y1["Write standalone OpenAI-style JSONL<br/>prompt, messages, tools, metadata"]
    D --> E["Resolve and load sources"]
    E --> F["Convert traces to messages + tools"]
    F --> G["Render tokenizer chat template"]
    G --> H["Collect typed supervision spans"]
    H --> I["Drop, trim, or report bad / oversized rows"]
    I --> J["Prepared Dataset<br/>text + teich_supervised_spans"]
    J --> K["SFTTrainer tokenization"]
    K --> L["mask_data"]
    L --> M["Align spans to token offsets"]
    M --> N["Create labels<br/>context = -100<br/>selected assistant/tool tokens = target ids"]
    N --> O["trainer.train"]
```

## Generation Flow

```mermaid
flowchart TD
    A["User runs teich generate -c config.yaml"] --> B["Load Config.from_yaml"]
    B --> C["Read prompt inputs"]
    C --> C1{"Prompt file type?"}
    C1 -->|"recommended"| C2["JSONL / NDJSON<br/>prompt, github_repo, follow_up_prompts"]
    C1 -->|"also supported"| C3["JSON, CSV, or plain text"]
    C2 --> D["Normalize PromptInput rows"]
    C3 --> D
    D --> E["Deduplicate by prompt + follow_up_prompts"]
    E --> F{"--resume?"}
    F -->|"yes"| F1["Scan completed output rows"]
    F1 --> F2["Skip completed prompt keys"]
    F -->|"no"| G["Use all pending prompt inputs"]
    F2 --> G

    G --> H{"agent.provider"}
    H -->|"codex"| I["Run Codex CLI in Docker"]
    H -->|"pi"| J["Run Pi agent in Docker"]
    H -->|"claude-code"| K["Run Claude Code in Docker"]
    H -->|"hermes"| L["Run Hermes Agent in Docker"]
    H -->|"chat"| M["Call OpenAI-compatible API directly"]

    I --> I1["Copy and normalize native Codex JSONL"]
    J --> J1["Copy and normalize native Pi JSONL"]
    K --> K1["Copy native Claude transcript JSONL"]
    L --> L1["Export Hermes state.db sessions"]
    M --> M1["Write structured chat rows"]

    I1 --> N["Copy sandbox snapshot"]
    J1 --> N
    K1 --> N
    L1 --> N
    M1 --> O["Write compact dataset README"]
    N --> O
    O --> P["Embed small tool schemas or write tools.json"]
    P --> Q{"publish.repo_id set?"}
    Q -->|"yes"| R["Upload JSONL + README/tools metadata to HF dataset repo"]
    Q -->|"no"| S["Leave local output ready for prepare_data"]
    R --> S
```

## `prepare_data()` Flow

```mermaid
flowchart TD
    A["prepare_data(source_or_dataset, tokenizer, ...)"] --> B["Resolve source"]
    B --> C{"Input type?"}
    C -->|"datasets.Dataset"| D["Use Dataset directly"]
    C -->|"str or Path"| E["load_traces"]
    C -->|"sequence"| F["Resolve each source"]
    C -->|"source mix"| G["Parse ratios and per-source options"]

    E --> E1{"Local path exists?"}
    E1 -->|"yes"| E2["Read local file / directory"]
    E1 -->|"no"| E3["Download HF dataset JSONL + README/tools snapshots"]
    E2 --> H["Convert traces to training rows"]
    E3 --> H
    F --> H
    G --> G1["Resolve each source independently"]
    G1 --> G2["Allocate row counts from requested ratios"]
    G2 --> H
    D --> I["Format rows"]
    H --> I

    I --> J["Validate messages and tools"]
    J --> K{"validate_tools=True?"}
    K -->|"yes"| K1["Check tool names and required args"]
    K -->|"no"| L["Render chat template"]
    K1 --> L
    L --> M["Inject invisible markers around candidate supervised fields"]
    M --> N["Collect typed character spans"]
    N --> O{"max_length set?"}
    O -->|"yes"| P["Measure rendered length"]
    P --> Q{"too long?"}
    Q -->|"drop"| R["Drop row"]
    Q -->|"trim_followups"| S["Trim final follow-up turns"]
    Q -->|"error"| T["Raise ValueError"]
    Q -->|"fits"| U["Keep row"]
    O -->|"no"| U
    S --> U
    U --> V{"tokenize=True?"}
    V -->|"yes"| W["Add input_ids + attention_mask"]
    V -->|"no"| X["Keep text + spans"]
    W --> Y["Return Dataset or (Dataset, PrepareReport)"]
    X --> Y
```

## `mask_data()` Flow

```mermaid
flowchart TD
    A["Create trainer with prepared dataset"] --> B["Trainer has text and/or input_ids"]
    B --> C["mask_data(trainer, ...)"]
    C --> D["Resolve tokenizer and text column"]
    D --> E{"packing enabled?"}
    E -->|"yes"| F["Raise ValueError"]
    E -->|"no"| G["Process train_dataset and eval_dataset"]
    G --> H{"input_ids present?"}
    H -->|"yes"| I["Use existing tokens"]
    H -->|"no, text present"| J["Tokenize text"]
    J --> I
    I --> K{"teich_supervised_spans present?"}
    K -->|"yes"| L["Retokenize with offsets and align spans"]
    K -->|"no"| M["Fallback infer assistant/tool spans from rendered text"]
    L --> N["Build labels"]
    M --> N
    N --> O{"all labels are -100?"}
    O -->|"yes"| P["Raise ValueError"]
    O -->|"no"| Q{"too many supervised tokens?"}
    Q -->|"yes"| R["Drop row"]
    Q -->|"no"| S["Keep input_ids + labels"]
    R --> T{"any rows left?"}
    S --> T
    T -->|"no"| U["Raise ValueError"]
    T -->|"yes"| V{"audit enabled?"}
    V -->|"yes"| W["Run SFT label audit"]
    V -->|"no"| X["Attach preview helper"]
    W --> X
    X --> Y["Return trainer"]
```

## Key Guarantee

```text
prepare_data keeps human-readable text plus typed span metadata.
mask_data converts the selected spans into exact token-level labels after trainer tokenization.
teich convert writes standalone OpenAI-style message JSONL without tokenizer rendering or token labels.
```

This lets Teich stay compatible with TRL / Unsloth trainer flows while still controlling exactly what the model learns.
