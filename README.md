# pi-agent-core

A stateful LLM agent framework for Python with tool execution, event streaming, steering/follow-up message queuing, and proxy transport.

## Installation

```bash
uv add pi-agent-core
```

Optional Anthropic adapter:

```bash
uv add "pi-agent-core[anthropic]"
```

## Overview

pi-agent-core provides a minimal, LLM-agnostic agent loop that handles orchestration between your application and any LLM provider. You bring your own streaming function — the library handles state management, tool execution, event dispatch, mid-turn steering, and follow-up queuing.

### Key Features

- **LLM-agnostic** — works with any provider via `StreamFn`
- **Real-time event streaming** — two-level event system for agent lifecycle and LLM streaming primitives
- **Tool execution** — define tools with JSON Schema parameters and async execute functions
- **Steering & follow-up queues** — interrupt mid-turn or queue messages for after completion
- **Cancellation** — cooperative cancellation via `asyncio.Event`
- **Proxy transport** — built-in SSE proxy client for routing through a backend server
- **Fully typed** — Pydantic models throughout with `py.typed` marker

## Quick Start

```python
import asyncio
from pi_agent_core import (
    Agent,
    AgentOptions,
    AgentEvent,
    AgentTool,
    AgentToolSchema,
    AgentToolResult,
    Model,
    TextContent,
)


# 1. Define your tools
async def greet(tool_call_id, params, cancel_event=None, on_update=None):
    name = params.get("name", "world")
    return AgentToolResult(content=[TextContent(text=f"Hello, {name}!")])


greet_tool = AgentTool(
    name="greet",
    description="Greet someone by name",
    parameters=AgentToolSchema(
        properties={"name": {"type": "string", "description": "Name to greet"}},
        required=["name"],
    ),
    execute=greet,
)


# 2. Implement a StreamFn for your LLM provider
# (see "Implementing a StreamFn" section below)
async def my_stream_fn(model, context, options):
    ...


# 3. Create and run the agent
agent = Agent(AgentOptions(stream_fn=my_stream_fn))
agent.set_model(Model(api="anthropic", provider="anthropic", id="claude-sonnet-4-20250514"))
agent.set_system_prompt("You are a helpful assistant.")
agent.set_tools([greet_tool])

# Subscribe to events
def on_event(event: AgentEvent):
    print(f"Event: {event.type}")

agent.subscribe(on_event)

# Send a prompt
asyncio.run(agent.prompt("Say hello to Alice"))
```

## Architecture

```
Agent                     ← High-level stateful wrapper, subscriptions, queues
    ↓
agent_loop()              ← Core orchestration: prompt → stream → tools → steering loop
    ↓
StreamFn (user-provided)  ← You implement LLM streaming integration
    or
stream_proxy()            ← Built-in SSE proxy client as a StreamFn
```

### Modules

| Module | Responsibility |
|---|---|
| `types.py` | All Pydantic models: content blocks, messages, events, tools, config, state, and `StreamResult` |
| `agent_loop.py` | `agent_loop()` and `agent_loop_continue()` async generators — streaming, tool execution, steering, follow-ups |
| `agent.py` | `Agent` class wrapping the loop with state management, event subscriptions, abort/reset, and queue management |
| `proxy.py` | `stream_proxy()` SSE client using httpx — reconstructs partial messages from server-stripped delta events |
| `anthropic.py` | Optional Anthropic Messages adapter (`stream_anthropic`) |

## Implementing a StreamFn

The library is LLM-agnostic. You provide a `stream_fn(model, context, options)` that returns a **procedural** `StreamResult` dict:

- `events`: `AsyncIterator[AssistantMessageEvent]`
- `result`: `Callable[[], Awaitable[AssistantMessage]]`

```python
from collections.abc import AsyncIterator

from pi_agent_core import (
    AssistantMessage,
    AssistantMessageEvent,
    StreamResult,
    StreamStartEvent,
    StreamTextStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
    StreamDoneEvent,
    TextContent,
    Model,
    AgentContext,
    SimpleStreamOptions,
)


async def my_stream_fn(
    model: Model,
    context: AgentContext,
    options: SimpleStreamOptions,
) -> StreamResult:
    partial = AssistantMessage(api=model.api, provider=model.provider, model=model.id)

    async def events_iter() -> AsyncIterator[AssistantMessageEvent]:
        # Example only — replace with provider events
        yield StreamStartEvent(partial=partial)

        partial.content = [TextContent(text="")]
        yield StreamTextStartEvent(content_index=0, partial=partial)

        partial.content[0].text += "Hello!"
        yield StreamTextDeltaEvent(content_index=0, delta="Hello!", partial=partial)
        yield StreamTextEndEvent(content_index=0, content="Hello!", partial=partial)

        partial.stop_reason = "stop"
        yield StreamDoneEvent(reason="stop", message=partial)

    async def result() -> AssistantMessage:
        return partial

    return {"events": events_iter(), "result": result}
```

## Event System

### Agent Events (10 types)

Covers agent lifecycle, turns, messages, and tool execution:

`agent_start`, `agent_end`, `turn_start`, `turn_end`, `message_start`, `message_update`, `message_end`, `tool_execution_start`, `tool_execution_update`, `tool_execution_end`

### Assistant Message Events (12 types)

Covers LLM streaming primitives consumed internally by the loop:

`start`, `text_start`, `text_delta`, `text_end`, `thinking_start`, `thinking_delta`, `thinking_end`, `toolcall_start`, `toolcall_delta`, `toolcall_end`, `done`, `error`

## Steering & Follow-up Queues

Steering messages interrupt the agent mid-turn (skipping remaining tool calls):

```python
agent.steer(UserMessage(content=[TextContent(text="Actually, use a different approach")]))
```

Follow-up messages trigger new turns after the current run completes:

```python
agent.follow_up(UserMessage(content=[TextContent(text="Now summarize the results")]))
```

Both support `"one-at-a-time"` (default) or `"all"` dequeue modes.

## Built-in Adapters

### Anthropic

```python
from pi_agent_core.anthropic import stream_anthropic

agent = Agent(AgentOptions(
    stream_fn=stream_anthropic,
))
```

Uses `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` automatically. You can also set `ANTHROPIC_BASE_URL`.

### Proxy Transport

For apps that route LLM calls through a backend server:

```python
from pi_agent_core import Agent, AgentOptions, stream_proxy, ProxyStreamOptions

agent = Agent(AgentOptions(
    stream_fn=lambda model, context, options: stream_proxy(
        model,
        context,
        ProxyStreamOptions(
            **options.model_dump(),
            auth_token="your-auth-token",
            proxy_url="https://your-proxy.example.com",
        ),
    ),
))
```

## Development

```bash
uv sync                     # Install dependencies
uv run pytest               # Run all tests
uv run pytest -v --tb=short # Verbose with short tracebacks
uv run ruff check .         # Lint
uv run ruff format .        # Format
```

## Credits

This is a Python port of the TypeScript [`@mariozechner/pi-agent-core`](https://github.com/badlogic/pi-mono) package from the **pi-mono** repository. The original TypeScript implementation by [Mario Zechner](https://github.com/mariozechner) provides the architecture, abstractions, and design that this library faithfully mirrors.

## License

[MIT](LICENSE)
