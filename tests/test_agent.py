"""Tests for Agent class."""

import pytest
from conftest import make_mock_stream_result

from pi_agent_core import (
    Agent,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    AgentToolSchema,
    Model,
    TextContent,
    ToolCall,
    UserMessage,
)


def make_model() -> Model:
    return Model(api="test", provider="test", id="test-model")


def make_stream_fn(text: str = "Hello!", tool_calls: list[ToolCall] | None = None):
    """Create a mock stream function that returns predefined text."""

    async def stream_fn(model, context, options):
        return make_mock_stream_result(text=text, tool_calls=tool_calls)

    return stream_fn


class TestAgentCreation:
    def test_default_creation(self):
        agent = Agent()
        assert agent.state.is_streaming is False
        assert agent.state.messages == []
        assert agent.state.thinking_level == "off"

    def test_with_options(self):
        model = make_model()
        agent = Agent(
            AgentOptions(
                initial_state={"system_prompt": "You are helpful.", "model": model},
                steering_mode="all",
            )
        )
        assert agent.state.system_prompt == "You are helpful."
        assert agent.state.model.id == "test-model"
        assert agent.get_steering_mode() == "all"

    def test_set_model(self):
        agent = Agent()
        model = make_model()
        agent.set_model(model)
        assert agent.state.model.id == "test-model"

    def test_set_system_prompt(self):
        agent = Agent()
        agent.set_system_prompt("Be helpful")
        assert agent.state.system_prompt == "Be helpful"


class TestAgentSubscription:
    def test_subscribe_and_unsubscribe(self):
        agent = Agent()
        events = []
        unsub = agent.subscribe(lambda e: events.append(e))
        assert callable(unsub)
        unsub()
        # After unsubscribe, no events should be received


class TestAgentQueues:
    def test_steering_queue(self):
        agent = Agent()
        msg = UserMessage(content=[TextContent(text="steer")])
        agent.steer(msg)
        assert agent.has_queued_messages()

    def test_follow_up_queue(self):
        agent = Agent()
        msg = UserMessage(content=[TextContent(text="follow up")])
        agent.follow_up(msg)
        assert agent.has_queued_messages()

    def test_clear_all_queues(self):
        agent = Agent()
        agent.steer(UserMessage(content=[TextContent(text="steer")]))
        agent.follow_up(UserMessage(content=[TextContent(text="follow")]))
        agent.clear_all_queues()
        assert not agent.has_queued_messages()

    def test_dequeue_one_at_a_time(self):
        agent = Agent(AgentOptions(steering_mode="one-at-a-time"))
        msg1 = UserMessage(content=[TextContent(text="steer1")])
        msg2 = UserMessage(content=[TextContent(text="steer2")])
        agent.steer(msg1)
        agent.steer(msg2)
        dequeued = agent._dequeue_steering_messages()
        assert len(dequeued) == 1
        assert dequeued[0] == msg1
        assert agent.has_queued_messages()

    def test_dequeue_all(self):
        agent = Agent(AgentOptions(steering_mode="all"))
        msg1 = UserMessage(content=[TextContent(text="steer1")])
        msg2 = UserMessage(content=[TextContent(text="steer2")])
        agent.steer(msg1)
        agent.steer(msg2)
        dequeued = agent._dequeue_steering_messages()
        assert len(dequeued) == 2
        assert not agent.has_queued_messages()


class TestAgentReset:
    def test_reset(self):
        agent = Agent()
        agent.append_message(UserMessage(content=[TextContent(text="hi")]))
        agent.steer(UserMessage(content=[TextContent(text="steer")]))
        agent.reset()
        assert agent.state.messages == []
        assert not agent.has_queued_messages()
        assert agent.state.is_streaming is False


@pytest.mark.asyncio
class TestAgentPrompt:
    async def test_prompt_string(self):
        model = make_model()
        agent = Agent(
            AgentOptions(
                initial_state={"model": model},
                stream_fn=make_stream_fn("Hello from agent!"),
            )
        )

        events = []
        agent.subscribe(lambda e: events.append(e))

        await agent.prompt("Hi there")

        assert not agent.state.is_streaming
        assert len(agent.state.messages) > 0

        # Should have agent_start, turn_start, message events, turn_end, agent_end
        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    async def test_prompt_while_streaming_raises(self):
        agent = Agent(
            AgentOptions(
                initial_state={"model": make_model()},
                stream_fn=make_stream_fn(),
            )
        )
        # Manually set streaming to simulate busy state
        agent._state.is_streaming = True

        with pytest.raises(RuntimeError, match="already processing"):
            await agent.prompt("test")

    async def test_prompt_no_model_raises(self):
        agent = Agent()
        with pytest.raises(ValueError, match="No model configured"):
            await agent.prompt("test")

    async def test_continue_no_messages_raises(self):
        agent = Agent(AgentOptions(initial_state={"model": make_model()}))
        with pytest.raises(ValueError, match="No messages"):
            await agent.continue_()

    async def test_wait_for_idle(self):
        model = make_model()
        agent = Agent(
            AgentOptions(
                initial_state={"model": model},
                stream_fn=make_stream_fn(),
            )
        )
        # When not running, wait_for_idle should resolve immediately
        await agent.wait_for_idle()


@pytest.mark.asyncio
class TestAgentToolExecution:
    async def test_tool_execution(self):
        model = make_model()

        async def execute_tool(tool_call_id, params, cancel_event=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(text=f"Result for {params.get('query', '')}")],
                details={"query": params.get("query", "")},
            )

        tool = AgentTool(
            name="search",
            description="Search for something",
            label="Search",
            parameters=AgentToolSchema(
                properties={"query": {"type": "string"}},
                required=["query"],
            ),
            execute=execute_tool,
        )

        tool_call = ToolCall(id="tc1", name="search", arguments={"query": "test"})

        # Create a stream function that returns a message with tool calls
        # followed by a final response
        call_count = 0

        async def stream_fn_with_tools(m, context, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_stream_result(text="Let me search.", tool_calls=[tool_call])
            else:
                return make_mock_stream_result(text="Found results!")

        agent = Agent(
            AgentOptions(
                initial_state={"model": model, "tools": [tool]},
                stream_fn=stream_fn_with_tools,
            )
        )

        events = []
        agent.subscribe(lambda e: events.append(e))

        await agent.prompt("Search for test")

        event_types = [e.type for e in events]
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types
