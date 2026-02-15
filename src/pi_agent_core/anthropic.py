"""
Anthropic streaming adapter for pi-agent-core.

Implements the StreamFn protocol using the official anthropic Python SDK,
converting between pi-agent-core types and Anthropic's native API format.

Requires the ``anthropic`` package: ``pip install pi-agent-core[anthropic]``
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

import anthropic

from .types import (
    AgentContext,
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamResult,
    StreamStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
    StreamTextStartEvent,
    StreamThinkingDeltaEvent,
    StreamThinkingEndEvent,
    StreamThinkingStartEvent,
    StreamToolCallDeltaEvent,
    StreamToolCallEndEvent,
    StreamToolCallStartEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "toolUse",
}


# ---------------------------------------------------------------------------
# Pure conversion functions
# ---------------------------------------------------------------------------


def _convert_user_content(content: TextContent | ImageContent) -> dict[str, Any]:
    """Convert a user content block to Anthropic format."""
    if isinstance(content, TextContent):
        return {"type": "text", "text": content.text}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": content.media_type,
            "data": content.data,
        },
    }


def _convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert AgentContext messages to Anthropic API format.

    Consecutive ToolResultMessages are merged into a single user message,
    as required by the Anthropic API.
    """
    api_messages: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, UserMessage):
            api_messages.append(
                {
                    "role": "user",
                    "content": [_convert_user_content(c) for c in msg.content],
                }
            )

        elif isinstance(msg, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, TextContent) and c.text:
                    blocks.append({"type": "text", "text": c.text})
                elif isinstance(c, ThinkingContent) and c.thinking:
                    block: dict[str, Any] = {"type": "thinking", "thinking": c.thinking}
                    if c.thinking_signature:
                        block["signature"] = c.thinking_signature
                    blocks.append(block)
                elif isinstance(c, ToolCall):
                    blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
            if blocks:
                api_messages.append({"role": "assistant", "content": blocks})

        elif isinstance(msg, ToolResultMessage):
            tool_result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": [_convert_user_content(c) for c in msg.content],
                "is_error": msg.is_error,
            }
            # Merge consecutive tool results into a single user message
            if api_messages and api_messages[-1].get("role") == "user" and api_messages[-1].get("_tool_results"):
                api_messages[-1]["content"].append(tool_result_block)
            else:
                api_messages.append({"role": "user", "content": [tool_result_block], "_tool_results": True})

    # Strip the internal marker before returning
    for msg_dict in api_messages:
        msg_dict.pop("_tool_results", None)

    return api_messages


def _convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert AgentTool definitions to Anthropic tool format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": {
                "type": t.parameters.type,
                "properties": t.parameters.properties,
                "required": t.parameters.required,
            },
        }
        for t in tools
    ]


def _build_request(model: Model, context: AgentContext, options: SimpleStreamOptions) -> dict[str, Any]:
    """Build the kwargs dict for ``client.messages.stream()``."""
    kwargs: dict[str, Any] = {
        "model": model.id,
        "messages": _convert_messages(context.messages),
        "max_tokens": options.max_tokens if options.max_tokens is not None else 8192,
    }

    if context.system_prompt:
        kwargs["system"] = context.system_prompt

    if options.temperature is not None:
        kwargs["temperature"] = options.temperature

    if context.tools:
        kwargs["tools"] = _convert_tools(context.tools)

    if options.reasoning and options.reasoning != "off":
        budget = 4096
        if options.thinking_budgets:
            budget = getattr(options.thinking_budgets, options.reasoning, budget)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

    return kwargs


def _create_client(options: SimpleStreamOptions) -> anthropic.AsyncAnthropic:
    """Create an AsyncAnthropic client from stream options."""
    api_key = options.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return anthropic.AsyncAnthropic(**client_kwargs)


def _map_event(
    event: Any,
    partial: AssistantMessage,
    block_types: dict[int, str],
    tool_json: dict[int, str],
) -> AssistantMessageEvent | None:
    """Map a single Anthropic streaming event to an AssistantMessageEvent.

    Mutates ``partial`` in place to accumulate message state.
    Returns None for unrecognised event types.
    """
    event_type = event.type

    if event_type == "message_start":
        if hasattr(event, "message") and hasattr(event.message, "usage"):
            partial.usage.input = getattr(event.message.usage, "input_tokens", 0)
        return StreamStartEvent(partial=partial)

    if event_type == "content_block_start":
        idx = event.index
        block = event.content_block

        while len(partial.content) <= idx:
            partial.content.append(TextContent())

        if block.type == "text":
            partial.content[idx] = TextContent()
            block_types[idx] = "text"
            return StreamTextStartEvent(content_index=idx, partial=partial)

        if block.type == "thinking":
            partial.content[idx] = ThinkingContent()
            block_types[idx] = "thinking"
            return StreamThinkingStartEvent(content_index=idx, partial=partial)

        if block.type == "tool_use":
            partial.content[idx] = ToolCall(id=block.id, name=block.name)
            block_types[idx] = "tool_use"
            tool_json[idx] = ""
            return StreamToolCallStartEvent(content_index=idx, partial=partial)

    elif event_type == "content_block_delta":
        idx = event.index
        delta = event.delta

        if delta.type == "text_delta":
            content = partial.content[idx]
            if isinstance(content, TextContent):
                content.text += delta.text
            return StreamTextDeltaEvent(content_index=idx, delta=delta.text, partial=partial)

        if delta.type == "thinking_delta":
            content = partial.content[idx]
            if isinstance(content, ThinkingContent):
                content.thinking += delta.thinking
            return StreamThinkingDeltaEvent(content_index=idx, delta=delta.thinking, partial=partial)

        if delta.type == "input_json_delta":
            content = partial.content[idx]
            if isinstance(content, ToolCall):
                if content.partial_json is None:
                    content.partial_json = ""
                content.partial_json += delta.partial_json
                tool_json[idx] = content.partial_json
                with contextlib.suppress(json.JSONDecodeError):
                    content.arguments = json.loads(content.partial_json)
            return StreamToolCallDeltaEvent(content_index=idx, delta=delta.partial_json, partial=partial)

    elif event_type == "content_block_stop":
        idx = event.index
        bt = block_types.get(idx)

        if bt == "text":
            content = partial.content[idx]
            return StreamTextEndEvent(
                content_index=idx,
                content=content.text if isinstance(content, TextContent) else "",
                partial=partial,
            )

        if bt == "thinking":
            content = partial.content[idx]
            return StreamThinkingEndEvent(
                content_index=idx,
                content=content.thinking if isinstance(content, ThinkingContent) else "",
                partial=partial,
            )

        if bt == "tool_use":
            content = partial.content[idx]
            if isinstance(content, ToolCall):
                accumulated = tool_json.get(idx, "")
                if accumulated:
                    with contextlib.suppress(json.JSONDecodeError):
                        content.arguments = json.loads(accumulated)
                content.partial_json = None
                return StreamToolCallEndEvent(content_index=idx, tool_call=content, partial=partial)

    elif event_type == "message_delta":
        if hasattr(event, "delta"):
            raw_reason = getattr(event.delta, "stop_reason", None)
            if isinstance(raw_reason, str):
                partial.stop_reason = STOP_REASON_MAP.get(raw_reason, "stop")
        if hasattr(event, "usage"):
            input_tokens = getattr(event.usage, "input_tokens", None)
            if input_tokens is not None:
                partial.usage.input = input_tokens
            partial.usage.output = getattr(event.usage, "output_tokens", 0)
            partial.usage.total_tokens = partial.usage.input + partial.usage.output
        return StreamDoneEvent(reason=partial.stop_reason, message=partial)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def stream_anthropic(
    model: Model,
    context: AgentContext,
    options: SimpleStreamOptions,
) -> StreamResult:
    """Stream function adapter for Anthropic's Messages API.

    Implements StreamFn using a procedural result object:
      - events: async iterator of AssistantMessageEvent
      - result: async callable returning final AssistantMessage
    """
    client = _create_client(options)
    request = _build_request(model, context, options)

    queue: asyncio.Queue[AssistantMessageEvent | None] = asyncio.Queue()
    done = asyncio.Event()
    state: dict[str, Any] = {"final": None, "task": None}

    partial = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
    block_types: dict[int, str] = {}
    tool_json: dict[int, str] = {}

    async def events_iter():
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    async def result() -> AssistantMessage:
        await done.wait()
        final = state["final"]
        if final is None:
            raise RuntimeError("No result available")
        return final

    async def _pump() -> None:
        nonlocal partial

        try:
            async with client.messages.stream(**request) as raw_stream:
                async for event in raw_stream:
                    if options.cancel_event and options.cancel_event.is_set():
                        raise RuntimeError("Request aborted by user")

                    mapped = _map_event(event, partial, block_types, tool_json)
                    if mapped is not None:
                        queue.put_nowait(mapped)

            if options.cancel_event and options.cancel_event.is_set():
                raise RuntimeError("Request aborted by user")

            state["final"] = partial

        except Exception as error:
            reason = "aborted" if (options.cancel_event and options.cancel_event.is_set()) else "error"
            partial.stop_reason = reason
            partial.error_message = str(error)
            queue.put_nowait(StreamErrorEvent(reason=reason, error=partial))
            state["final"] = partial

        finally:
            done.set()
            queue.put_nowait(None)

    state["task"] = asyncio.create_task(_pump())

    return {"events": events_iter(), "result": result}
