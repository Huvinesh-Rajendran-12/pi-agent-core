# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest                          # Run all 25 tests
uv run pytest tests/test_agent.py      # Run agent tests only
uv run pytest -k test_tool_execution   # Run a single test by name
uv run pytest -v --tb=short            # Verbose with short tracebacks
uv run ruff check .                    # Lint (pycodestyle, pyflakes, isort, bugbear, simplify)
uv run ruff format .                   # Auto-format
uv sync                               # Install / sync dependencies
```

## Architecture

This is a Python port of the TypeScript `@mariozechner/pi-agent-core` package — a stateful LLM agent framework with tool execution, event streaming, and message queuing.

### Layer Diagram

```
agent.py (Agent)          ← High-level stateful wrapper, subscriptions, queues
    ↓
agent_loop.py             ← Core orchestration: prompt→stream→tools→steering loop
    ↓
StreamFn (user-provided)  ← User implements their own LLM streaming integration
    or
proxy.py (stream_proxy)   ← SSE proxy client as a StreamFn implementation
```

### Key Abstractions

**StreamFn protocol** — The library is LLM-agnostic. Users provide a `stream_fn(model, context, options) -> StreamResult` that yields `AssistantMessageEvent` objects and implements `async result() -> AssistantMessage`. The proxy module provides one implementation; users write their own for direct LLM calls.

**Message transformation pipeline** — Messages are stored as `AgentMessage` (extensible) throughout, and converted to LLM-compatible `Message[]` only at the call boundary via the user-provided `convert_to_llm` callback.

**Two-level event system** — `AgentEvent` (10 types) covers agent lifecycle, turns, messages, and tool execution. `AssistantMessageEvent` (12 types) covers LLM streaming primitives (text/thinking/toolcall deltas). Agent subscribers receive `AgentEvent`; the loop internally consumes `AssistantMessageEvent` from the stream.

**Steering & follow-up queues** — Steering messages interrupt mid-turn (skip remaining tool calls). Follow-up messages trigger new turns after completion. Both use async callbacks with "one-at-a-time" or "all" dequeue modes.

### Module Responsibilities

- **types.py** — All Pydantic models: content blocks, messages, events, tool definitions, config, state, and the `StreamResult` protocol
- **agent_loop.py** — `agent_loop()` and `agent_loop_continue()` as async generators. Handles streaming, tool execution with cancellation, steering interruption, and follow-up continuation
- **agent.py** — `Agent` class wrapping the loop with state management, event subscription, abort/reset, and queue management. `continue_()` uses trailing underscore to avoid keyword conflict
- **proxy.py** — `stream_proxy()` SSE client using httpx. Reconstructs partial messages from server-stripped delta events. Uses `asyncio.Queue` internally to bridge push-based SSE to async iteration

### Cancellation

Uses `asyncio.Event` (not `AbortSignal` from TS). The Agent creates a cancel event per run; `abort()` sets it. The loop and tools check it cooperatively.

## Style Guide

Enforced by **ruff** (`uv run ruff check .` / `uv run ruff format .`). Each rule maps back to a Zen principle:

| Zen aphorism | Ruff rule set | What it enforces |
|---|---|---|
| "Readability counts" | `E`, `W` (pycodestyle) | Whitespace, line length ≤ 120, PEP 8 |
| "Errors should never pass silently" | `F` (pyflakes), `B` (bugbear) | Unused imports/vars, common bugs, mutable defaults |
| "There should be one obvious way" | `I` (isort) | Deterministic import order |
| "Beautiful is better than ugly" | `UP` (pyupgrade) | Modern syntax (`X \| Y` unions, f-strings, etc.) |
| "Simple is better than complex" | `SIM` (simplify) | Simplifiable `if`/`return`/`with` patterns |
| Miscellaneous quality | `RUF` | Ruff-specific checks |

Conventions:
- **Line length**: 120 characters max
- **Quotes**: double (`"`) — enforced by `ruff format`
- **Imports**: stdlib → third-party → first-party (`pi_agent_core`), one import per line for `from` imports
- **Type annotations**: use `X | Y` for new code; `Union[X, Y]` is allowed in Pydantic model fields
- **`__init__.py`**: F401 (unused import) is ignored — re-exports are intentional

## The Zen of Python (PEP 20)

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```
