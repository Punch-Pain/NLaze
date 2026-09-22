# NLaze — Neural Lazy Terminal

## Vision

**Type intent, not syntax.**

The terminal is the fastest interface ever built — but it punishes you for one wrong
letter. `sl` instead of `ls`. `docments` instead of `documents`. `git statts` instead
of `git status`. One keystroke wrong, whole command dead.

LLMs fixed this in editors and chat. Claude renames folders. Copilot writes functions.
But the terminal — the tool developers spend the most time in — is still dumb.

**NLaze makes the terminal understand what you meant, not just what you typed.**

It uses [Laya](https://huggingface.co/convaiinnovations/laya), a 421M-parameter
decision model running locally at 12ms latency, to:

1. **Fix typos seamlessly** — `cd docments` → `cd documents`, auto-run if unambiguous
2. **Resolve ambiguity intelligently** — two similar folders? Laya picks based on context
3. **Translate natural language** — "rename x to y" → `mv x y`, no memorizing flags
4. **Learn your patterns** — records your corrections, gets smarter over time

## Principles

- **Correct commands are free.** Valid commands execute directly, zero overhead, zero
  Laya inference. The model only activates when something is wrong or unclear.
- **12ms is invisible.** Laya's latency on consumer GPUs is imperceptible. Fixing a
  typo costs less than human reaction time.
- **Local, private, offline.** Everything runs on your GPU. No cloud, no API keys,
  no telemetry. Your commands never leave your machine.
- **Built on battle-tested foundations.** zoxide (39k stars, MIT) for path resolution.
  Laya (Apache 2.0) for intent. We don't reinvent matching algorithms.
- **Destructive commands require confirmation.** `rm -rf`, `git push --force`, etc.
  always get a confirm prompt. Safety first.

## What NLaze is NOT

- Not an LLM chatbot in your terminal (that's what Claude Code is)
- Not a shell replacement (works WITH fish, bash, zsh)
- Not a cloud service (everything local)
- Not a GUI (terminal-native, works in any terminal)

## The three modes

| Mode | Trigger | Example | Latency |
|---|---|---|---|
| **Direct** | Valid command | `ls -la` | 0ms (no Laya) |
| **Fix** | Typo / wrong command | `sl` → `ls` | 12ms |
| **Intent** | Natural language | "rename x to y" → `mv x y` | 12ms |

## Success looks like

You type `ddoc` and land in `~/documents` before you'd have finished backspacing
to fix it yourself. You type "show me what changed" and `git diff` runs. You never
think about it — it just works.
