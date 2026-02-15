"""Tests for agent_loop and agent_loop_continue."""

import pytest
from conftest import make_mock_stream_result

from pi_agent_core import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    AgentToolSchema,
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    agent_loop,
    agent_loop_continue,
)


def make_model() -> Model:
    return Model(api="test", provider="test", id="test-model")


def _default_convert(msgs):
    return [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")]


def make_config(model: Model | None = None) -> AgentLoopConfig:
    return AgentLoopConfig(
        model=model or make_model(),
        convert_to_llm=_default_convert,
    )


def make_stream_fn(text: str = "Hello!", tool_calls: list[ToolCall] | None = None):
    async def stream_fn(model, context, options):
        return make_mock_stream_result(text=text, tool_calls=tool_calls)

    return stream_fn


@pytest.mark.asyncio
class TestAgentLoop:
    async def test_basic_loop(self):
        """Test a simple prompt-response cycle."""
        model = make_model()
        config = make_config(model)
        context = AgentContext(system_prompt="Be helpful", messages=[], tools=[])
        prompt = UserMessage(content=[TextContent(text="Hi")])

        events = []
        async for event in agent_loop(
            [prompt],
            context,
            config,
            stream_fn=make_stream_fn("Hello!"),
        ):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "message_start" in event_types
        assert "message_end" in event_types
        assert "turn_start" in event_types
        assert "turn_end" in event_types
        assert "agent_end" in event_types

    async def test_loop_event_order(self):
        """Test that events come in the expected order."""
        config = make_config()
        context = AgentContext(system_prompt="", messages=[], tools=[])
        prompt = UserMessage(content=[TextContent(text="Hi")])

        events = []
        async for event in agent_loop(
            [prompt],
            context,
            config,
            stream_fn=make_stream_fn("Response"),
        ):
            events.append(event)

        types = [e.type for e in events]
        # Should start with agent_start, turn_start
        assert types[0] == "agent_start"
        assert types[1] == "turn_start"
        # Should end with agent_end
        assert types[-1] == "agent_end"

    async def test_loop_with_tool_calls(self):
        """Test that tool calls are executed properly."""

        async def execute_tool(tool_call_id, params, cancel_event=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(text="tool result")],
                details={},
            )

        tool = AgentTool(
            name="test_tool",
            description="A test tool",
            label="Test",
            parameters=AgentToolSchema(),
            execute=execute_tool,
        )

        tool_call = ToolCall(id="tc1", name="test_tool", arguments={})
        call_count = 0

        async def stream_fn(model, context, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_stream_result(text="Using tool", tool_calls=[tool_call])
            return make_mock_stream_result(text="Done")

        config = make_config()
        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        prompt = UserMessage(content=[TextContent(text="Use tool")])

        events = []
        async for event in agent_loop([prompt], context, config, stream_fn=stream_fn):
            events.append(event)

        types = [e.type for e in events]
        assert "tool_execution_start" in types
        assert "tool_execution_end" in types


@pytest.mark.asyncio
class TestAgentLoopContinue:
    async def test_continue_from_tool_result(self):
        """Test continuing from a tool result message."""
        config = make_config()
        tool_result = ToolResultMessage(
            tool_call_id="tc1",
            tool_name="test",
            content=[TextContent(text="result")],
        )
        context = AgentContext(
            system_prompt="",
            messages=[
                UserMessage(content=[TextContent(text="Hi")]),
                tool_result,
            ],
            tools=[],
        )

        events = []
        async for event in agent_loop_continue(
            context,
            config,
            stream_fn=make_stream_fn("Continued response"),
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "agent_start" in types
        assert "agent_end" in types

    async def test_continue_empty_context_raises(self):
        """Test that continuing with empty context raises."""
        config = make_config()
        context = AgentContext(system_prompt="", messages=[], tools=[])

        with pytest.raises(ValueError, match="no messages"):
            async for _ in agent_loop_continue(context, config, stream_fn=make_stream_fn()):
                pass

    async def test_continue_from_assistant_raises(self):
        """Test that continuing from an assistant message raises."""
        config = make_config()
        context = AgentContext(
            system_prompt="",
            messages=[AssistantMessage(content=[TextContent(text="hi")])],
            tools=[],
        )

        with pytest.raises(ValueError, match="assistant"):
            async for _ in agent_loop_continue(context, config, stream_fn=make_stream_fn()):
                pass


@pytest.mark.asyncio
class TestSteeringMessages:
    async def test_steering_skips_remaining_tools(self):
        """Test that steering messages skip remaining tool calls."""
        steering_call_count = 0

        async def get_steering():
            nonlocal steering_call_count
            steering_call_count += 1
            if steering_call_count == 2:
                # Return steering on the second call (before second tool execution).
                # Call 1: initial check at start of _run_loop (returns []).
                # Call 2: before tool 2 in _execute_tool_calls (returns steering).
                return [UserMessage(content=[TextContent(text="Stop!")])]
            return []

        async def execute_tool(tool_call_id, params, cancel_event=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")], details={})

        tool = AgentTool(
            name="slow_tool",
            description="A slow tool",
            label="Slow",
            parameters=AgentToolSchema(),
            execute=execute_tool,
        )

        tool_calls = [
            ToolCall(id="tc1", name="slow_tool", arguments={}),
            ToolCall(id="tc2", name="slow_tool", arguments={}),
        ]

        call_count = 0

        async def stream_fn(model, context, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_stream_result(text="Using tools", tool_calls=tool_calls)
            return make_mock_stream_result(text="Done after steering")

        config = AgentLoopConfig(
            model=make_model(),
            convert_to_llm=_default_convert,
            get_steering_messages=get_steering,
        )
        context = AgentContext(system_prompt="", messages=[], tools=[tool])
        prompt = UserMessage(content=[TextContent(text="Do things")])

        events = []
        async for event in agent_loop([prompt], context, config, stream_fn=stream_fn):
            events.append(event)

        # The second tool call should be skipped
        tool_end_events = [e for e in events if e.type == "tool_execution_end"]
        assert len(tool_end_events) == 2  # Both reported, but second is skipped
        assert tool_end_events[1].is_error  # Skipped tool is marked as error


@pytest.mark.asyncio
class TestFollowUpMessages:
    async def test_follow_up_continues_loop(self):
        """Test that follow-up messages continue the agent loop."""
        follow_up_returned = False

        async def get_follow_up():
            nonlocal follow_up_returned
            if not follow_up_returned:
                follow_up_returned = True
                return [UserMessage(content=[TextContent(text="Follow up")])]
            return []

        config = AgentLoopConfig(
            model=make_model(),
            convert_to_llm=_default_convert,
            get_follow_up_messages=get_follow_up,
        )
        context = AgentContext(system_prompt="", messages=[], tools=[])
        prompt = UserMessage(content=[TextContent(text="Start")])

        events = []
        async for event in agent_loop([prompt], context, config, stream_fn=make_stream_fn("Response")):
            events.append(event)

        # Should have multiple turn_start events (original + follow-up)
        turn_starts = [e for e in events if e.type == "turn_start"]
        assert len(turn_starts) >= 2
