"""
Type definitions for pi-agent-core.

Mirrors the TypeScript types.ts, using Pydantic models and Python typing constructs.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Content types (mirrors TextContent, ImageContent, ThinkingContent, ToolCall)
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    media_type: str  # e.g. "image/png"
    data: str  # base64-encoded


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None


class ToolCall(BaseModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = {}
    partial_json: str | None = None


# Union of all content block types
ContentBlock = TextContent | ImageContent | ThinkingContent | ToolCall


# ---------------------------------------------------------------------------
# Message types (mirrors Message union: UserMessage, AssistantMessage, ToolResultMessage)
# ---------------------------------------------------------------------------


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent] = []
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class UsageCost(BaseModel):
    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_write: float = 0
    total: float = 0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


StopReason = Literal["stop", "length", "toolUse", "aborted", "error"]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = []
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ToolResultMessage(BaseModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent] = []
    details: Any = None
    is_error: bool = False
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


# Base LLM message type
Message = UserMessage | AssistantMessage | ToolResultMessage

# AgentMessage is the same as Message.
AgentMessage = Message


# ---------------------------------------------------------------------------
# Model definition (mirrors pi-ai Model)
# ---------------------------------------------------------------------------


class Model(BaseModel):
    """LLM model identifier."""

    api: str
    provider: str
    id: str


# ---------------------------------------------------------------------------
# Thinking level
# ---------------------------------------------------------------------------

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

Transport = Literal["sse", "websocket"]


# ---------------------------------------------------------------------------
# Tool types
# ---------------------------------------------------------------------------


class AgentToolResult(BaseModel):
    """Result returned by a tool execution."""

    content: list[TextContent | ImageContent] = []
    details: Any = None


# Callback type for streaming tool execution updates
AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class AgentToolSchema(BaseModel):
    """JSON Schema description for tool parameters."""

    type: str = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


class AgentTool(BaseModel):
    """
    Agent tool definition with execution function.

    The `execute` callable signature:
        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    label: str = ""
    parameters: AgentToolSchema = Field(default_factory=AgentToolSchema)
    execute: Callable[..., Awaitable[AgentToolResult]] = Field(exclude=True)


# ---------------------------------------------------------------------------
# Agent context
# ---------------------------------------------------------------------------


class AgentContext(BaseModel):
    """Context passed to the agent loop."""

    model_config = {"arbitrary_types_allowed": True}

    system_prompt: str = ""
    messages: list[Message] = []
    tools: list[AgentTool] = []


# ---------------------------------------------------------------------------
# Agent loop config
# ---------------------------------------------------------------------------


class ThinkingBudgets(BaseModel):
    """Token budgets for thinking levels (provider-specific)."""

    minimal: int = 1024
    low: int = 2048
    medium: int = 4096
    high: int = 8192
    xhigh: int = 16384


class AgentLoopConfig(BaseModel):
    """Configuration for the agent loop."""

    model_config = {"arbitrary_types_allowed": True}

    model: Model
    reasoning: ThinkingLevel | None = None
    session_id: str | None = None
    transport: Transport = "sse"
    thinking_budgets: ThinkingBudgets | None = None
    max_retry_delay_ms: int | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    convert_to_llm: Callable[[list[Message]], list[Message] | Awaitable[list[Message]]] = Field(exclude=True)
    transform_context: Callable[[list[Message], asyncio.Event | None], Awaitable[list[Message]]] | None = Field(
        default=None, exclude=True
    )
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = Field(default=None, exclude=True)
    get_steering_messages: Callable[[], Awaitable[list[Message]]] | None = Field(default=None, exclude=True)
    get_follow_up_messages: Callable[[], Awaitable[list[Message]]] | None = Field(default=None, exclude=True)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Agent state containing all configuration and conversation data."""

    model_config = {"arbitrary_types_allowed": True}

    system_prompt: str = ""
    model: Model = Field(default_factory=lambda: Model(api="", provider="", id=""))
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = []
    messages: list[Message] = []
    is_streaming: bool = False
    stream_message: Message | None = None
    pending_tool_calls: set[str] = Field(default_factory=set)
    error: str | None = None


# ---------------------------------------------------------------------------
# Agent events
# ---------------------------------------------------------------------------


class AgentStartEvent(BaseModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[Message] = []


class TurnStartEvent(BaseModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    type: Literal["turn_end"] = "turn_end"
    message: Message | None = None
    tool_results: list[ToolResultMessage] = []


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: Message | None = None


class MessageUpdateEvent(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: Message | None = None
    assistant_message_event: Any = None  # AssistantMessageEvent from stream


class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: Message | None = None


class ToolExecutionStartEvent(BaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any = None


class ToolExecutionUpdateEvent(BaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any = None
    partial_result: Any = None


class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)


# ---------------------------------------------------------------------------
# Assistant message event types (for streaming from LLM / proxy)
# ---------------------------------------------------------------------------


class StreamStartEvent(BaseModel):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class StreamTextStartEvent(BaseModel):
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class StreamTextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamTextEndEvent(BaseModel):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class StreamThinkingStartEvent(BaseModel):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class StreamThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamThinkingEndEvent(BaseModel):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class StreamToolCallStartEvent(BaseModel):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class StreamToolCallDeltaEvent(BaseModel):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamToolCallEndEvent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class StreamDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    reason: StopReason
    message: AssistantMessage


class StreamErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    reason: StopReason
    error: AssistantMessage


AssistantMessageEvent = (
    StreamStartEvent
    | StreamTextStartEvent
    | StreamTextDeltaEvent
    | StreamTextEndEvent
    | StreamThinkingStartEvent
    | StreamThinkingDeltaEvent
    | StreamThinkingEndEvent
    | StreamToolCallStartEvent
    | StreamToolCallDeltaEvent
    | StreamToolCallEndEvent
    | StreamDoneEvent
    | StreamErrorEvent
)


# ---------------------------------------------------------------------------
# Stream function type
# ---------------------------------------------------------------------------

# StreamFn: an async callable that takes (model, context, config) and returns
# a procedural stream response:
#   - events: async iterator of AssistantMessageEvent
#   - result: async callable returning the final AssistantMessage


class StreamResult(TypedDict):
    events: AsyncIterator[AssistantMessageEvent]
    result: Callable[[], Awaitable[AssistantMessage]]


StreamFn = Callable[..., Awaitable[StreamResult] | StreamResult]


# ---------------------------------------------------------------------------
# Simple stream options (base for config)
# ---------------------------------------------------------------------------


class SimpleStreamOptions(BaseModel):
    """Options passed to the stream function."""

    model_config = {"arbitrary_types_allowed": True}

    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: ThinkingLevel | None = None
    session_id: str | None = None
    transport: Transport = "sse"
    thinking_budgets: ThinkingBudgets | None = None
    max_retry_delay_ms: int | None = None
    cancel_event: asyncio.Event | None = Field(default=None, exclude=True)
