"""Shared test fixtures and helpers."""

from pi_agent_core import (
    AssistantMessage,
    StreamDoneEvent,
    StreamStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
    StreamTextStartEvent,
    TextContent,
    ToolCall,
)


class MockStreamResult:
    """Mock stream result that yields predefined events."""

    def __init__(self, text: str = "Hello!", tool_calls: list[ToolCall] | None = None):
        self._text = text
        self._tool_calls = tool_calls or []
        self._events = []
        self._final: AssistantMessage | None = None
        self._build_events()
        self._index = 0

    def _build_events(self):
        content_blocks = []
        content_blocks.append(TextContent(text=self._text))
        content_blocks.extend(self._tool_calls)

        self._final = AssistantMessage(
            content=content_blocks,
            api="test",
            provider="test",
            model="test-model",
            stop_reason="toolUse" if self._tool_calls else "stop",
        )

        partial = AssistantMessage(api="test", provider="test", model="test-model")
        self._events.append(StreamStartEvent(partial=partial))

        partial_with_text = AssistantMessage(
            content=[TextContent(text="")],
            api="test",
            provider="test",
            model="test-model",
        )
        self._events.append(StreamTextStartEvent(content_index=0, partial=partial_with_text))
        self._events.append(
            StreamTextDeltaEvent(
                content_index=0,
                delta=self._text,
                partial=partial_with_text,
            )
        )
        self._events.append(
            StreamTextEndEvent(
                content_index=0,
                content=self._text,
                partial=partial_with_text,
            )
        )
        self._events.append(StreamDoneEvent(reason=self._final.stop_reason, message=self._final))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def result(self) -> AssistantMessage:
        return self._final
