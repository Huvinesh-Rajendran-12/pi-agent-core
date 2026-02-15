"""
Agent loop that works with AgentMessage throughout.
Transforms to Message[] only at the LLM call boundary.

Mirrors agent-loop.ts from the TypeScript implementation.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    SimpleStreamOptions,
    StreamFn,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
)


async def agent_loop(
    prompts: list[Message],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Start an agent loop with new prompt messages.
    The prompts are added to the context and events are emitted for them.

    Yields AgentEvent instances.
    """
    new_messages: list[Message] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=list(context.tools),
    )

    yield AgentStartEvent()
    yield TurnStartEvent()

    for prompt in prompts:
        yield MessageStartEvent(message=prompt)
        yield MessageEndEvent(message=prompt)

    async for event in _run_loop(current_context, new_messages, config, cancel_event, stream_fn):
        yield event


async def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Continue an agent loop from the current context without adding a new message.
    Used for retries - context already has user message or tool results.
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    last_msg = context.messages[-1]
    last_role = last_msg.role if hasattr(last_msg, "role") else last_msg.get("role")
    if last_role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[Message] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=list(context.tools),
    )

    yield AgentStartEvent()
    yield TurnStartEvent()

    async for event in _run_loop(current_context, new_messages, config, cancel_event, stream_fn):
        yield event


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[Message],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    stream_fn: StreamFn | None,
) -> AsyncGenerator[AgentEvent, None]:
    """Main loop logic shared by agent_loop and agent_loop_continue."""
    first_turn = True
    # Check for steering messages at start (matches TS runLoop behavior)
    pending_messages: list[Message] = []
    if config.get_steering_messages:
        pending_messages = await config.get_steering_messages()

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or len(pending_messages) > 0:
            if not first_turn:
                yield TurnStartEvent()
            else:
                first_turn = False

            # Process pending messages
            if pending_messages:
                for message in pending_messages:
                    yield MessageStartEvent(message=message)
                    yield MessageEndEvent(message=message)
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            assistant_msg: AssistantMessage | None = None
            async for event in _stream_assistant_response(current_context, config, cancel_event, stream_fn):
                if isinstance(event, MessageEndEvent):
                    assistant_msg = event.message
                yield event

            if assistant_msg is None:
                # Should not happen, but bail out
                yield AgentEndEvent(messages=new_messages)
                return

            new_messages.append(assistant_msg)

            if assistant_msg.stop_reason in ("error", "aborted"):
                yield TurnEndEvent(message=assistant_msg, tool_results=[])
                yield AgentEndEvent(messages=new_messages)
                return

            # Check for tool calls
            tool_calls = [c for c in assistant_msg.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            steering_after_tools: list[Message] | None = None
            if has_more_tool_calls:
                async for tool_result, events, steering in _execute_tool_calls(
                    current_context.tools,
                    assistant_msg,
                    cancel_event,
                    config.get_steering_messages,
                ):
                    # Yield events immediately so subscribers can react (e.g., steering)
                    for evt in events:
                        yield evt

                    # Yield control to event loop so steering can be processed
                    await asyncio.sleep(0)

                    tool_results.append(tool_result)
                    current_context.messages.append(tool_result)
                    new_messages.append(tool_result)

                    if steering and steering_after_tools is None:
                        steering_after_tools = steering

            yield TurnEndEvent(message=assistant_msg, tool_results=tool_results)

            # Get steering messages after turn
            if steering_after_tools and len(steering_after_tools) > 0:
                pending_messages = steering_after_tools
                steering_after_tools = None
            elif config.get_steering_messages:
                pending_messages = await config.get_steering_messages()
            else:
                pending_messages = []

        # Check for follow-up messages
        follow_up_messages: list[Message] = []
        if config.get_follow_up_messages:
            follow_up_messages = await config.get_follow_up_messages()

        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        break

    yield AgentEndEvent(messages=new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    stream_fn: StreamFn | None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Stream an assistant response from the LLM.
    Transforms AgentMessage[] to Message[] at the LLM call boundary.
    """
    # Apply context transform if configured
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(messages, cancel_event)

    # Convert to LLM-compatible messages
    convert_result = config.convert_to_llm(messages)
    if inspect.isawaitable(convert_result):
        llm_messages = await convert_result
    else:
        llm_messages = convert_result

    # Build LLM context
    llm_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    if stream_fn is None:
        raise ValueError("No stream function provided. You must supply a stream_fn.")

    # Resolve API key
    resolved_api_key = config.api_key
    if config.get_api_key:
        key_result = config.get_api_key(config.model.provider)
        if inspect.isawaitable(key_result):
            key_result = await key_result
        if key_result:
            resolved_api_key = key_result

    # Call stream function
    stream_options = SimpleStreamOptions(
        api_key=resolved_api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        reasoning=config.reasoning,
        session_id=config.session_id,
        transport=config.transport,
        thinking_budgets=config.thinking_budgets,
        max_retry_delay_ms=config.max_retry_delay_ms,
        cancel_event=cancel_event,
    )

    response = stream_fn(config.model, llm_context, stream_options)
    if inspect.isawaitable(response):
        response = await response

    if not isinstance(response, dict) or "events" not in response or "result" not in response:
        raise TypeError("StreamFn must return {'events': AsyncIterator, 'result': async callable}")

    events = response["events"]
    get_result = response["result"]

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in events:
        event_type = event.type

        if event_type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            yield MessageStartEvent(message=_copy_msg(partial_message))

        elif event_type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                yield MessageUpdateEvent(
                    message=_copy_msg(partial_message),
                    assistant_message_event=event,
                )

        elif event_type in ("done", "error"):
            final_message = await get_result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                yield MessageStartEvent(message=_copy_msg(final_message))
            yield MessageEndEvent(message=final_message)
            return

    # Fallback: get result if loop ended without done/error
    final_message = await get_result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
    if not added_partial:
        yield MessageStartEvent(message=_copy_msg(final_message))
    yield MessageEndEvent(message=final_message)


def _copy_msg(msg: Message) -> Message:
    """Create a shallow copy of a message."""
    if hasattr(msg, "model_copy"):
        return msg.model_copy()
    return msg


async def _execute_tool_calls(
    tools: list[AgentTool],
    assistant_message: AssistantMessage,
    cancel_event: asyncio.Event | None,
    get_steering_messages: Any | None,
) -> AsyncIterator[tuple[ToolResultMessage, list[AgentEvent], list[Message] | None]]:
    """
    Execute tool calls from an assistant message.
    Yields after each tool: (tool_result, events, steering_messages).
    Steering_messages is non-None if the tool execution should stop early.
    """
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]

    tools_by_name: dict[str, AgentTool] = {}
    for t in tools:
        tools_by_name.setdefault(t.name, t)

    for index, tool_call in enumerate(tool_calls):
        # Check steering BEFORE executing each tool (after the first).
        # In TS, steering is checked after each tool via push-based streams where
        # subscribers react synchronously. In Python's async generator model, events
        # are yielded to the caller between iterations, so checking before the next
        # tool is the equivalent timing — subscribers have already seen previous events.
        if index > 0 and get_steering_messages:
            steering = await get_steering_messages()
            if steering:
                # Skip this and all remaining tool calls
                for skipped in tool_calls[index:]:
                    skip_result = _skip_tool_call(skipped)
                    yield skip_result["tool_result"], skip_result["events"], steering
                return

        tool = tools_by_name.get(tool_call.name)

        events: list[AgentEvent] = [
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        ]

        result: AgentToolResult
        is_error = False

        try:
            if tool is None:
                raise ValueError(f"Tool {tool_call.name} not found")

            # Validate arguments against tool schema (basic validation)
            validated_args = _validate_tool_arguments(tool_call)

            update_events: list[AgentEvent] = []

            def on_update(
                partial_result: AgentToolResult,
                _events: list[AgentEvent] = update_events,
                _tc: ToolCall = tool_call,
            ) -> None:
                _events.append(
                    ToolExecutionUpdateEvent(
                        tool_call_id=_tc.id,
                        tool_name=_tc.name,
                        args=_tc.arguments,
                        partial_result=partial_result,
                    )
                )

            result = await tool.execute(
                tool_call.id,
                validated_args,
                cancel_event,
                on_update,
            )
            events.extend(update_events)

        except Exception as e:
            result = AgentToolResult(
                content=[TextContent(text=str(e))],
                details={},
            )
            is_error = True

        events.append(
            ToolExecutionEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                is_error=is_error,
            )
        )

        tool_result_message = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            details=result.details,
            is_error=is_error,
        )

        events.append(MessageStartEvent(message=tool_result_message))
        events.append(MessageEndEvent(message=tool_result_message))

        yield tool_result_message, events, None


def _validate_tool_arguments(tool_call: ToolCall) -> dict[str, Any]:
    """Basic validation of tool arguments. Returns the arguments dict."""
    args = tool_call.arguments
    if not isinstance(args, dict):
        if isinstance(args, str):
            args = json.loads(args)
        else:
            raise ValueError(
                f"Tool {tool_call.name}: expected dict or JSON string arguments, got {type(args).__name__}"
            )
    return args


def _skip_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Create a skipped tool call result."""
    result = AgentToolResult(
        content=[TextContent(text="Skipped due to queued user message.")],
        details={},
    )

    events: list[AgentEvent] = [
        ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ),
        ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=True,
        ),
    ]

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details={},
        is_error=True,
    )

    events.append(MessageStartEvent(message=tool_result_message))
    events.append(MessageEndEvent(message=tool_result_message))

    return {"tool_result": tool_result_message, "events": events}
