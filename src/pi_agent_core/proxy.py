"""
Proxy stream function for apps that route LLM calls through a server.
The server manages auth and proxies requests to LLM providers.

Mirrors proxy.ts from the TypeScript implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import httpx

from .types import (
    AgentContext,
    AssistantMessage,
    AssistantMessageEvent,
    Model,
    SimpleStreamOptions,
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
    Usage,
    UsageCost,
)


class ProxyStreamOptions(SimpleStreamOptions):
    """Options for the proxy stream function."""

    auth_token: str
    proxy_url: str


# Backward-compat export name.
ProxyAsyncStream = StreamResult


async def stream_proxy(
    model: Model,
    context: AgentContext,
    options: ProxyStreamOptions,
) -> StreamResult:
    """
    Stream function that proxies through a server instead of calling LLM providers directly.

    Returns a procedural stream result:
      - events: async iterator of AssistantMessageEvent
      - result: async callable returning the final AssistantMessage
    """
    queue: asyncio.Queue[AssistantMessageEvent | None] = asyncio.Queue()
    done = asyncio.Event()
    state: dict[str, Any] = {"final": None, "task": None}

    partial = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
    )

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

    async def _run() -> None:
        nonlocal partial

        try:
            body = {
                "model": model.model_dump(),
                "context": {
                    "systemPrompt": context.system_prompt,
                    "messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in context.messages],
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "label": t.label,
                            "parameters": t.parameters.model_dump(),
                        }
                        for t in context.tools
                    ]
                    if context.tools
                    else [],
                },
                "options": {
                    "temperature": options.temperature,
                    "maxTokens": options.max_tokens,
                    "reasoning": options.reasoning,
                },
            }

            headers = {
                "Authorization": f"Bearer {options.auth_token}",
                "Content-Type": "application/json",
            }

            async with (
                httpx.AsyncClient(timeout=httpx.Timeout(None)) as client,
                client.stream(
                    "POST",
                    f"{options.proxy_url}/api/stream",
                    headers=headers,
                    json=body,
                ) as response,
            ):
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                    try:
                        error_data = json.loads(error_text)
                        if "error" in error_data:
                            error_message = f"Proxy error: {error_data['error']}"
                    except (json.JSONDecodeError, KeyError):
                        pass
                    raise RuntimeError(error_message)

                buffer = ""
                async for chunk in response.aiter_text():
                    if options.cancel_event and options.cancel_event.is_set():
                        raise RuntimeError("Request aborted by user")

                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data:
                                proxy_event = json.loads(data)
                                event = _process_proxy_event(proxy_event, partial)
                                if event is not None:
                                    queue.put_nowait(event)

            if options.cancel_event and options.cancel_event.is_set():
                raise RuntimeError("Request aborted by user")

            state["final"] = partial

        except Exception as error:
            error_message = str(error)
            reason = "aborted" if (options.cancel_event and options.cancel_event.is_set()) else "error"
            partial.stop_reason = reason
            partial.error_message = error_message
            queue.put_nowait(
                StreamErrorEvent(
                    reason=reason,
                    error=partial,
                )
            )
            state["final"] = partial

        finally:
            done.set()
            queue.put_nowait(None)

    state["task"] = asyncio.create_task(_run())

    return {"events": events_iter(), "result": result}


def _process_proxy_event(
    proxy_event: dict[str, Any],
    partial: AssistantMessage,
) -> AssistantMessageEvent | None:
    """Process a proxy event and update the partial message."""
    event_type = proxy_event.get("type")

    if event_type == "start":
        return StreamStartEvent(partial=partial)

    elif event_type == "text_start":
        idx = proxy_event["contentIndex"]
        # Extend content list if needed
        while len(partial.content) <= idx:
            partial.content.append(TextContent())
        partial.content[idx] = TextContent()
        return StreamTextStartEvent(content_index=idx, partial=partial)

    elif event_type == "text_delta":
        idx = proxy_event["contentIndex"]
        content = partial.content[idx]
        if isinstance(content, TextContent):
            content.text += proxy_event["delta"]
            return StreamTextDeltaEvent(
                content_index=idx,
                delta=proxy_event["delta"],
                partial=partial,
            )
        raise RuntimeError("Received text_delta for non-text content")

    elif event_type == "text_end":
        idx = proxy_event["contentIndex"]
        content = partial.content[idx]
        if isinstance(content, TextContent):
            content.text_signature = proxy_event.get("contentSignature")
            return StreamTextEndEvent(
                content_index=idx,
                content=content.text,
                partial=partial,
            )
        raise RuntimeError("Received text_end for non-text content")

    elif event_type == "thinking_start":
        idx = proxy_event["contentIndex"]
        while len(partial.content) <= idx:
            partial.content.append(TextContent())
        partial.content[idx] = ThinkingContent()
        return StreamThinkingStartEvent(content_index=idx, partial=partial)

    elif event_type == "thinking_delta":
        idx = proxy_event["contentIndex"]
        content = partial.content[idx]
        if isinstance(content, ThinkingContent):
            content.thinking += proxy_event["delta"]
            return StreamThinkingDeltaEvent(
                content_index=idx,
                delta=proxy_event["delta"],
                partial=partial,
            )
        raise RuntimeError("Received thinking_delta for non-thinking content")

    elif event_type == "thinking_end":
        idx = proxy_event["contentIndex"]
        content = partial.content[idx]
        if isinstance(content, ThinkingContent):
            content.thinking_signature = proxy_event.get("contentSignature")
            return StreamThinkingEndEvent(
                content_index=idx,
                content=content.thinking,
                partial=partial,
            )
        raise RuntimeError("Received thinking_end for non-thinking content")

    elif event_type == "toolcall_start":
        idx = proxy_event["contentIndex"]
        while len(partial.content) <= idx:
            partial.content.append(TextContent())
        partial.content[idx] = ToolCall(
            id=proxy_event["id"],
            name=proxy_event["toolName"],
        )
        return StreamToolCallStartEvent(content_index=idx, partial=partial)

    elif event_type == "toolcall_delta":
        idx = proxy_event["contentIndex"]
        content = partial.content[idx]
        if isinstance(content, ToolCall):
            if content.partial_json is None:
                content.partial_json = ""
            content.partial_json += proxy_event["delta"]
            # Try to parse partial JSON
            with contextlib.suppress(json.JSONDecodeError):
                content.arguments = json.loads(content.partial_json)
            return StreamToolCallDeltaEvent(
                content_index=idx,
                delta=proxy_event["delta"],
                partial=partial,
            )
        raise RuntimeError("Received toolcall_delta for non-toolCall content")

    elif event_type == "toolcall_end":
        idx = proxy_event["contentIndex"]
        content = partial.content[idx]
        if isinstance(content, ToolCall):
            content.partial_json = None
            return StreamToolCallEndEvent(
                content_index=idx,
                tool_call=content,
                partial=partial,
            )
        return None

    elif event_type == "done":
        usage_data = proxy_event.get("usage", {})
        partial.stop_reason = proxy_event["reason"]
        partial.usage = _parse_usage(usage_data)
        stream_result = StreamDoneEvent(
            reason=proxy_event["reason"],
            message=partial,
        )
        return stream_result

    elif event_type == "error":
        usage_data = proxy_event.get("usage", {})
        partial.stop_reason = proxy_event["reason"]
        partial.error_message = proxy_event.get("errorMessage")
        partial.usage = _parse_usage(usage_data)
        return StreamErrorEvent(
            reason=proxy_event["reason"],
            error=partial,
        )

    return None


def _parse_usage(data: dict[str, Any]) -> Usage:
    """Parse usage data from proxy response."""
    cost_data = data.get("cost", {})
    return Usage(
        input=data.get("input", 0),
        output=data.get("output", 0),
        cache_read=data.get("cacheRead", 0),
        cache_write=data.get("cacheWrite", 0),
        total_tokens=data.get("totalTokens", 0),
        cost=UsageCost(
            input=cost_data.get("input", 0),
            output=cost_data.get("output", 0),
            cache_read=cost_data.get("cacheRead", 0),
            cache_write=cost_data.get("cacheWrite", 0),
            total=cost_data.get("total", 0),
        ),
    )
