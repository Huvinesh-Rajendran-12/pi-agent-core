"""
Integration tests for pi-agent-core using the Anthropic adapter.

These tests require a valid ANTHROPIC_API_KEY environment variable.
They are skipped when the key is not available.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run pytest tests/test_integration.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

from pi_agent_core import (
    Agent,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    AgentToolSchema,
    Model,
    TextContent,
    UserMessage,
)
from pi_agent_core.anthropic import stream_anthropic

# ---------------------------------------------------------------------------
# Skip all tests when no API key is available
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not API_KEY, reason="OPENROUTER_API_KEY or ANTHROPIC_API_KEY not set"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL = Model(api="anthropic-messages", provider="anthropic", id="anthropic/claude-haiku-4.5")


def make_agent(**overrides) -> Agent:
    """Create an agent pre-configured with the Anthropic adapter."""
    opts = {
        "initial_state": {"model": MODEL, "system_prompt": "You are a helpful assistant. Be concise."},
        "stream_fn": stream_anthropic,
    }
    opts.update(overrides)
    return Agent(AgentOptions(**opts))


def collect_events(agent: Agent) -> list:
    """Subscribe to agent events and return the list they're collected into."""
    events: list = []
    agent.subscribe(lambda e: events.append(e))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicPrompt:
    async def test_simple_text_response(self):
        """Agent should return a non-empty text response."""
        agent = make_agent()
        events = collect_events(agent)

        await agent.prompt("What is 2 + 2? Reply with just the number.")

        assert not agent.state.is_streaming
        assert len(agent.state.messages) >= 2  # user + assistant

        # Last message should be an assistant response with text
        last_msg = agent.state.messages[-1]
        assert last_msg.role == "assistant"
        text_blocks = [c for c in last_msg.content if isinstance(c, TextContent)]
        assert len(text_blocks) > 0
        assert "4" in text_blocks[0].text

        # Events should include full lifecycle
        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "message_start" in event_types
        assert "message_update" in event_types
        assert "message_end" in event_types
        assert "turn_end" in event_types
        assert "agent_end" in event_types

    async def test_streaming_events_contain_deltas(self):
        """Streaming events should contain text deltas."""
        agent = make_agent()
        events = collect_events(agent)

        await agent.prompt("Say 'hello world'")

        update_events = [e for e in events if e.type == "message_update"]
        assert len(update_events) > 0

        # At least one update should have a text_delta assistant_message_event
        has_text_delta = any(
            hasattr(e, "assistant_message_event")
            and e.assistant_message_event is not None
            and getattr(e.assistant_message_event, "type", None) == "text_delta"
            for e in update_events
        )
        assert has_text_delta


class TestMultiTurn:
    async def test_two_turn_conversation(self):
        """Agent should maintain context across two turns."""
        agent = make_agent()

        await agent.prompt("My name is Alice. Remember it.")
        await agent.prompt("What is my name?")

        assert len(agent.state.messages) >= 4  # user, assistant, user, assistant

        last_msg = agent.state.messages[-1]
        text_blocks = [c for c in last_msg.content if isinstance(c, TextContent)]
        combined_text = " ".join(b.text for b in text_blocks).lower()
        assert "alice" in combined_text


class TestToolExecution:
    async def test_single_tool_call(self):
        """Agent should call a tool and use the result."""

        async def execute_calculator(tool_call_id, params, cancel_event=None, on_update=None):
            expression = params.get("expression", "")
            try:
                result = str(eval(expression))
            except Exception as e:
                result = f"Error: {e}"
            return AgentToolResult(
                content=[TextContent(text=result)],
                details={"expression": expression},
            )

        calculator = AgentTool(
            name="calculator",
            description="Evaluate a mathematical expression. Input should be a valid Python math expression.",
            label="Calculator",
            parameters=AgentToolSchema(
                properties={"expression": {"type": "string", "description": "The math expression to evaluate"}},
                required=["expression"],
            ),
            execute=execute_calculator,
        )

        agent = make_agent(
            initial_state={"model": MODEL, "system_prompt": "Use the calculator tool.", "tools": [calculator]}
        )
        events = collect_events(agent)

        await agent.prompt("What is 137 * 456?")

        event_types = [e.type for e in events]
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types

        # The final response should contain the correct answer (allow for formatting)
        last_msg = agent.state.messages[-1]
        text_blocks = [c for c in last_msg.content if isinstance(c, TextContent)]
        combined_text = " ".join(b.text for b in text_blocks)
        assert "62,472" in combined_text or "62472" in combined_text

    async def test_tool_with_update_callback(self):
        """Tool execution update events should be emitted when on_update is called."""

        async def execute_with_updates(tool_call_id, params, cancel_event=None, on_update=None):
            if on_update:
                on_update(AgentToolResult(content=[TextContent(text="Working...")], details={}))
            return AgentToolResult(
                content=[TextContent(text="Done: result is 42")],
                details={},
            )

        tool = AgentTool(
            name="slow_task",
            description="A task that provides progress updates",
            label="Slow Task",
            parameters=AgentToolSchema(
                properties={"input": {"type": "string"}},
                required=["input"],
            ),
            execute=execute_with_updates,
        )

        agent = make_agent(
            initial_state={"model": MODEL, "system_prompt": "Use the slow_task tool for any request.", "tools": [tool]}
        )
        events = collect_events(agent)

        await agent.prompt("Run the slow task with input 'test'")

        event_types = [e.type for e in events]
        assert "tool_execution_start" in event_types
        assert "tool_execution_update" in event_types
        assert "tool_execution_end" in event_types


class TestAbort:
    async def test_abort_cancels_streaming(self):
        """Aborting the agent should stop streaming."""
        agent = make_agent()
        collect_events(agent)

        async def abort_after_delay():
            await asyncio.sleep(0.5)
            agent.abort()

        task = asyncio.create_task(abort_after_delay())

        await agent.prompt("Write a very long essay about the history of mathematics, at least 5000 words.")

        # Agent should have stopped — either with an error or aborted state
        assert not agent.state.is_streaming
        assert agent.state.error is not None
        assert task.done()


class TestFollowUp:
    async def test_follow_up_triggers_new_turn(self):
        """Follow-up messages should trigger additional turns."""
        agent = make_agent()
        events = collect_events(agent)

        # Queue a follow-up before prompting
        follow_up_msg = UserMessage(content=[TextContent(text="Now say goodbye.")])
        agent.follow_up(follow_up_msg)

        await agent.prompt("Say hello.")

        # Should have at least 4 messages: user("hello"), assistant, user("goodbye"), assistant
        assert len(agent.state.messages) >= 4

        turn_starts = [e for e in events if e.type == "turn_start"]
        assert len(turn_starts) >= 2


class TestSteering:
    async def test_steering_interrupts_tool_execution(self):
        """Steering messages should skip remaining tool calls."""
        call_count = 0

        async def slow_execute(tool_call_id, params, cancel_event=None, on_update=None):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return AgentToolResult(content=[TextContent(text=f"Result {call_count}")], details={})

        tool = AgentTool(
            name="slow_tool",
            description="A slow tool",
            label="Slow",
            parameters=AgentToolSchema(properties={"input": {"type": "string"}}, required=["input"]),
            execute=slow_execute,
        )

        agent = make_agent(
            initial_state={
                "model": MODEL,
                "system_prompt": "Always call slow_tool twice: once with input 'a' and once with input 'b'.",
                "tools": [tool],
            }
        )
        events = collect_events(agent)

        # Steer after the first tool execution completes
        steer_sent = False

        def on_event(e):
            nonlocal steer_sent
            if e.type == "tool_execution_end" and not steer_sent:
                steer_sent = True
                agent.steer(UserMessage(content=[TextContent(text="Stop, no more tools.")]))

        agent.subscribe(on_event)

        await agent.prompt("Call slow_tool twice")

        # The second tool call should have been skipped (is_error=True)
        tool_end_events = [e for e in events if e.type == "tool_execution_end"]
        if len(tool_end_events) >= 2:
            assert tool_end_events[-1].is_error


class TestUsageTracking:
    async def test_usage_is_populated(self):
        """The assistant message should have usage information."""
        agent = make_agent()

        await agent.prompt("Hi")

        assistant_msgs = [m for m in agent.state.messages if m.role == "assistant"]
        assert len(assistant_msgs) > 0

        usage = assistant_msgs[0].usage
        assert usage.input > 0
        assert usage.output > 0
        assert usage.total_tokens > 0


class TestErrorHandling:
    async def test_invalid_model_returns_error(self):
        """Using a nonexistent model should result in an error message."""
        bad_model = Model(api="anthropic-messages", provider="anthropic", id="nonexistent-model-xyz")
        agent = Agent(
            AgentOptions(
                initial_state={"model": bad_model, "system_prompt": "Hi"},
                stream_fn=stream_anthropic,
            )
        )

        await agent.prompt("Hello")

        assert agent.state.error is not None
        last_msg = agent.state.messages[-1]
        assert last_msg.role == "assistant"
        assert last_msg.stop_reason in ("error", "aborted")
