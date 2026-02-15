# pi-agent-core

A stateful LLM agent framework for Python with tool execution, event streaming, steering/follow-up message queuing, and proxy transport.

## Installation

```bash
uv add pi-agent-core
```

## Overview

pi-agent-core provides a minimal, LLM-agnostic agent loop that handles the orchestration between your application and any LLM provider. You bring your own streaming function — the library handles state management, tool execution, event dispatch, mid-turn steering, and follow-up queuing.

### Key Features

- **LLM-agnostic** — works with any provider via the `StreamFn` protocol
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
| `types.py` | All Pydantic models: content blocks, messages, events, tools, config, state, and the `StreamResult` protocol |
| `agent_loop.py` | `agent_loop()` and `agent_loop_continue()` async generators — streaming, tool execution, steering, follow-ups |
| `agent.py` | `Agent` class wrapping the loop with state management, event subscriptions, abort/reset, and queue management |
| `proxy.py` | `stream_proxy()` SSE client using httpx — reconstructs partial messages from server-stripped delta events |

## Implementing a StreamFn

The library is LLM-agnostic. You provide a `stream_fn(model, context, options)` that returns a `StreamResult` — an async iterator of `AssistantMessageEvent` objects with an `async result()` method.

```python
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


class MyStream:
    """Implements the StreamResult protocol."""

    def __init__(self):
        self._events = []
        self._final = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> AssistantMessageEvent:
        ...  # yield events from your LLM provider

    async def result(self) -> AssistantMessage:
        return self._final


async def my_stream_fn(
    model: Model,
    context: AgentContext,
    options: SimpleStreamOptions,
) -> StreamResult:
    stream = MyStream()
    # Start your LLM call, push events into the stream
    return stream
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

## Proxy Transport

For apps that route LLM calls through a backend server:

```python
from pi_agent_core import Agent, AgentOptions, stream_proxy, ProxyStreamOptions

agent = Agent(AgentOptions(
    stream_fn=lambda model, context, options: stream_proxy(
        model, context,
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

This is a Python port of the TypeScript [`@mariozechner/pi-agent-core`](https://github.com/nichochar/pi-mono) package from the **pi-mono** repository. The original TypeScript implementation by [Mario Zechner](https://github.com/mariozechner) provides the architecture, abstractions, and design that this library faithfully mirrors.

## License

[MIT](LICENSE)
