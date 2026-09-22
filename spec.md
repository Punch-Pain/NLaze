# NLaze — Technical Specification

## Overview

NLaze is a smart terminal assistant that uses the Laya decision model to fix typos,
resolve ambiguous paths, and translate natural language into shell commands. It layers
on top of existing shells without replacing them.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  User Input                      │
│          (fish / bash / zsh prompt)             │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            Shell Wrapper (fish hook)            │
│                                                  │
│  1. Validate command exists?                    │
│     ├─ YES → execute directly (0ms)             │
│     └─ NO  → capture input, send to daemon      │
└──────────────────────┬──────────────────────────┘
                       │ (socket / stdin)
                       ▼
┌─────────────────────────────────────────────────┐
│            NLaze Daemon (resident)              │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ 1. Candidate Generation                   │   │
│  │    • Edit distance (typo fixing)          │   │
│  │    • zoxide frecency lookup (paths)       │   │
│  │    • Keyword match (NL intent)            │   │
│  │    • $PATH scan (command names)           │   │
│  └──────────────────┬───────────────────────┘   │
│                     │ candidates                │
│  ┌──────────────────▼───────────────────────┐   │
│  │ 2. Laya Inference (12ms)                  │   │
│  │    • choice: pick best command/path       │   │
│  │    • score: confidence of match           │   │
│  │    • noul: is this destructive?           │   │
│  └──────────────────┬───────────────────────┘   │
│                     │ decision                   │
│  ┌──────────────────▼───────────────────────┐   │
│  │ 3. Action Router                          │   │
│  │    • confidence ≥ threshold → auto-run    │   │
│  │    • ambiguous → show options (numbered)  │   │
│  │    • destructive → require confirm        │   │
│  └──────────────────┬───────────────────────┘   │
│                     │                           │
│  ┌──────────────────▼───────────────────────┐   │
│  │ 4. Learning Store (SQLite)                │   │
│  │    • log: input → chosen correction       │   │
│  │    • boost: frequently chosen corrections │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Components

### 1. Shell Wrapper (`init.fish`)

Fish shell integration. Intercepts commands before execution.

```fish
# Pseudocode - fish event hook
function __nlaze_preexec --on-event fish_preexec
    set -l cmd $argv

    # Fast path: valid command → do nothing, let it run
    if __nlaze_validate $cmd
        return 0
    end

    # Slow path: send to daemon, get fix
    set -l fix (echo $cmd | nc -U /tmp/nlaze.sock)
    if test -n "$fix"
        echo "nlaze → $fix"
        eval $fix
    end
end
```

**Validation logic (zero cost):**
- First word exists in `$PATH`? → valid command
- `cd` target exists? → valid path
- Known builtin? (`cd`, `export`, `alias`, etc.) → valid
- Otherwise → send to daemon

### 2. Daemon (`nlaze-daemon`)

Python process, stays resident. Loads Laya once at startup.

**Socket:** Unix domain socket at `/tmp/nlaze.sock`
**Protocol:** Newline-delimited JSON

Request:
```json
{
  "input": "cd docments",
  "cwd": "/home/user/projects",
  "shell": "fish",
  "mode": "auto"
}
```

Response:
```json
{
  "action": "auto_run",
  "command": "cd documents",
  "confidence": 0.94,
  "latency_ms": 12
}
```

Or for ambiguous:
```json
{
  "action": "show_options",
  "options": [
    {"command": "cd documents", "score": 0.72},
    {"command": "cd docs", "score": 0.68}
  ],
  "latency_ms": 12
}
```

Or for destructive:
```json
{
  "action": "confirm",
  "command": "rm -rf node_modules",
  "reason": "destructive command",
  "confidence": 0.91,
  "latency_ms": 12
}
```

### 3. Candidate Generation

Three strategies, run in order, first match wins:

**a) Edit distance (typo fixing)**
- Levenshtein distance ≤ 2 on first word
- Sources: `$PATH` binaries, shell aliases, builtins
- Example: `sl` → `ls` (distance 1), `statts` → `status` (distance 2)

**b) Zoxide frecency (path resolution)**
- Delegate to `zoxide query <input>`
- Returns ranked directory matches based on usage frequency
- Example: `cd docments` → zoxide finds `~/documents` (frecency: high)

**c) Keyword match (NL intent)**
- Extract keywords from input
- Match against a command knowledge base (built into Laya fine-tuning data)
- Example: "rename x to y" → keywords: `rename`, `move`, `file` → candidates: `mv`, `git mv`, `rename`

### 4. NLaze Inference (custom fine-tuned model)

Model: fine-tuned from `convaiinnovations/laya` (ModernBERT-large, 421M params)
Shipped as quantized version (Q8 or Q4, chosen after benchmarking).
FP16 reference checkpoint also released.

Context: 512 tokens (English base), extendable via fine-tuning.

Questions schema:
```python
questions = {
    "fix": {
        "type": "choice",
        "instructions": "Which command did the user mean to type?",
        "criteria": {
            "option_0": "<candidate 1 full command>",
            "option_1": "<candidate 2 full command>",
            # ... up to ~15 candidates
        }
    },
    "destructive": {
        "type": "noul",
        "instructions": "Is this command destructive (irreversible data loss)?"
    }
}
```

**Thresholds:**
- Auto-run: confidence ≥ 0.85 AND not destructive
- Show options: 0.50 ≤ confidence < 0.85, OR multiple candidates within 0.15 of each other
- Confirm: destructive = true OR confidence < 0.50
- No match: zero candidates generated → report "command not found"

### 5. Learning Store

SQLite database at `~/.local/share/nlaze/history.db`

```sql
CREATE TABLE corrections (
    id INTEGER PRIMARY KEY,
    timestamp REAL,
    input TEXT NOT NULL,           -- what user typed
    chosen TEXT NOT NULL,          -- what was executed
    confidence REAL,
    source TEXT,                   -- 'auto' | 'user_selected'
    cwd TEXT
);

CREATE TABLE patterns (
    input_hash TEXT PRIMARY KEY,   -- normalized input
    chosen TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    last_used REAL
);
```

**Usage:**
- On startup: load top patterns into memory for boosting
- On correction: insert into corrections, upsert patterns
- On candidate generation: boost candidates that match known patterns
- Pattern lookup is O(1) hash match, adds <0.1ms

### 6. Destructive Command List

Commands always requiring confirmation:
```
rm -rf, rm -fr, mkfs, dd, shred, git push --force,
git reset --hard, git clean -fd, chmod -R 777,
sudo rm, > /dev/sd*, mv /dev/null, kill -9 1,
shutdown, reboot, halt, init 0, fork bomb (:(){ :|:& };:)
```

Configurable at `~/.config/nlaze/config.toml`.

## File Structure

```
NLaze/
├── vision.md
├── spec.md
├── README.md              # later
├── pyproject.toml
├── nlaze/
│   ├── __init__.py
│   ├── daemon.py          # resident daemon, socket server
│   ├── validate.py        # fast-path command validation
│   ├── candidates.py      # candidate generation (edit dist, zoxide, keywords)
│   ├── inference.py       # Laya model loading and predict
│   ├── router.py          # action routing (auto-run / show / confirm)
│   ├── store.py           # SQLite learning store
│   ├── config.py          # config loading
│   └── destructive.py     # destructive command patterns
├── shell/
│   ├── init.fish          # fish shell integration
│   ├── init.bash          # bash integration (later)
│   └── init.zsh           # zsh integration (later)
├── data/
│   └── commands.json      # command knowledge base for NL intent
├── tests/
│   ├── test_validate.py
│   ├── test_candidates.py
│   ├── test_router.py
│   └── test_store.py
└── scripts/
    └── install.sh
```

## Config

`~/.config/nlaze/config.toml`:
```toml
[general]
auto_run_threshold = 0.85
mode = "auto"                # auto | always_confirm | never_confirm

[model]
checkpoint = "Punch-Pain/nlaze-base"    # our fine-tuned checkpoint
version = "fp16"                        # fp16 | q8 | q4 (default chosen after benchmark)
device = "cuda"
max_candidates = 15

[learning]
enabled = true
db_path = "~/.local/share/nlaze/history.db"

[destructive]
extra_patterns = [
    "my-custom-dangerous-command --flag",
]
```

## Performance Targets

| Metric | Target |
|---|---|
| Valid command overhead | 0ms (no model, no socket) |
| Fix latency (end-to-end) | < 30ms (12ms model + 5ms socket + 10ms eval) |
| Daemon startup (no model) | < 100ms (socket only) |
| Model load (on terminal open) | < 5s (one-time) |
| Memory (resident, model loaded) | < 400MB VRAM (default quantized version) |
| Memory (resident, no model) | < 20MB RAM (socket only) |
| Candidate generation | < 2ms |
| Pattern lookup | < 0.1ms |

## Model Strategy

**We do not use stock Laya. We fine-tune our own.**

- Base: `convaiinnovations/laya` (ModernBERT-large, 421M params, FP16)
- Fine-tune on command correction + NL intent data
- Benchmark all quantization levels after training
- Release all versions, pick default based on benchmarks

### Quantization plan

| Version | Params | VRAM (approx) | Quality target |
|---|---|---|---|
| FP16 | 421M | ~840MB | 100% (reference) |
| Q8 | 421M | ~421MB | ~99.9% |
| Q4 | 421M | ~210MB | ~99% |
| FP16 base (ModernBERT-base) | 149M | ~298MB | fallback if needed |

Default is chosen after benchmarking, not assumed.

### VRAM budget

- Target: **under 400MB** for the shipped default
- Q8 or Q4 on the full model fits this
- Model loads only when a compatible terminal opens
- Model unloads when last terminal closes

## Phase Plan

### Phase 1 — Fine-tune the model (current)
- Generate training data (typos, NL intents, command pairs)
- Format as Laya's RLCD training schema
- Fine-tune full FP16 Laya on RTX 5050
- Evaluate accuracy on held-out test set
- Document benchmark results

### Phase 2 — Quantize & benchmark
- Export FP16 checkpoint
- Produce Q8, Q4 quantized versions
- Benchmark accuracy vs latency for each
- Benchmark VRAM usage for each
- Pick default based on data
- Release all versions

### Phase 3 — Daemon + integration
- Unix socket daemon (resident, model lazy-loaded)
- Fish shell wrapper (fast-path validation)
- Zoxide integration for path resolution
- SQLite learning store
- Destructive command detection
- Auto-run / show options / confirm logic

### Phase 4 — NL intent + polish
- Command knowledge base
- Natural language input handling
- Bash + zsh support
- Terminal-specific integration (Ghostty/Kitty)
- Public release
