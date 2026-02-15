"""Shared test fixtures and helpers."""

from collections.abc import AsyncIterator

from pi_agent_core import (
    AssistantMessage,
    StreamDoneEvent,
    StreamResult,
    StreamStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
    StreamTextStartEvent,
    TextContent,
    ToolCall,
)


def make_mock_stream_result(text: str = "Hello!", tool_calls: list[ToolCall] | None = None) -> StreamResult:
    """Create a procedural mock stream result for tests."""
    calls = tool_calls or []

    content_blocks = [TextContent(text=text), *calls]
    final = AssistantMessage(
        content=content_blocks,
        api="test",
        provider="test",
        model="test-model",
        stop_reason="toolUse" if calls else "stop",
    )

    partial = AssistantMessage(api="test", provider="test", model="test-model")
    partial_with_text = AssistantMessage(
        content=[TextContent(text="")],
        api="test",
        provider="test",
        model="test-model",
    )

    events = [
        StreamStartEvent(partial=partial),
        StreamTextStartEvent(content_index=0, partial=partial_with_text),
        StreamTextDeltaEvent(content_index=0, delta=text, partial=partial_with_text),
        StreamTextEndEvent(content_index=0, content=text, partial=partial_with_text),
        StreamDoneEvent(reason=final.stop_reason, message=final),
    ]

    async def events_iter() -> AsyncIterator:
        for event in events:
            yield event

    async def result() -> AssistantMessage:
        return final

    return {"events": events_iter(), "result": result}
