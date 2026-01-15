# Custom Agent Implementation Guide - Google ADK

## Overview

This guide shows how to implement custom agents by extending `BaseAgent`. Custom agents enable complex orchestration logic, conditional execution, state management, and dynamic agent selection beyond what standard workflow agents provide.

## Core Implementation Requirements

### Method Signature

You **must** implement one or both of these methods:

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    """Core logic for text-based conversation."""
    # Your implementation here
    yield Event(...)  # Must yield at least one Event

async def _run_live_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    """Core logic for video/audio-based conversation."""
    # Your implementation here
    yield Event(...)  # Must yield at least one Event
```

## Understanding InvocationContext

The `InvocationContext` provides access to:

- **Session State**: `ctx.session.state` - Dictionary for passing data between stages
- **Event History**: `ctx._get_events(current_invocation=True, current_branch=True)` - Access conversation history
- **Agent Reference**: `ctx.agent` - The current agent instance
- **Pause Signals**: `ctx.should_pause_invocation(event)` - Check if execution should pause
- **State Management**: `ctx.set_agent_state(agent_name, end_of_agent=True)` - Update agent state
- **Control Flow**: `ctx.end_invocation` - Signal to end execution

## Learning from LlmAgent Implementation

The `LlmAgent._run_async_impl` shows the pattern:

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    # 1. Load agent state (for resumability)
    agent_state = self._load_agent_state(ctx, BaseAgentState)

    # 2. Check if resuming a sub-agent transfer
    if agent_state is not None and (
        agent_to_transfer := self._get_subagent_to_resume(ctx)
    ):
        # Run the sub-agent and yield its events
        async with Aclosing(agent_to_transfer.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # Mark this agent as complete
        ctx.set_agent_state(self.name, end_of_agent=True)
        yield self._create_agent_state_event(ctx)
        return

    # 3. Execute main agent logic (in LlmAgent, this is the LLM flow)
    should_pause = False
    async with Aclosing(self._llm_flow.run_async(ctx)) as agen:
        async for event in agen:
            self.__maybe_save_output_to_state(event)  # Save output to state
            yield event
            if ctx.should_pause_invocation(event):
                should_pause = True

    # 4. Check for pause conditions
    if should_pause:
        return

    # 5. Yield final state event (for resumability)
    if ctx.is_resumable:
        events = ctx._get_events(current_invocation=True, current_branch=True)
        if events and any(ctx.should_pause_invocation(e) for e in events[-2:]):
            return
        ctx.set_agent_state(self.name, end_of_agent=True)
        yield self._create_agent_state_event(ctx)
```

### Key Patterns from LlmAgent

1. **Use `Aclosing`** for async generator cleanup
2. **Save outputs to state** using `ctx.session.state[key] = value`
3. **Yield events immediately** for responsive streaming
4. **Handle pauses properly** by checking `ctx.should_pause_invocation(event)`
5. **Mark completion** with `ctx.set_agent_state(self.name, end_of_agent=True)`

## Implementing Sub-Agent Invocation

### Basic Pattern

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    # Run a sub-agent
    async with Aclosing(self.my_sub_agent.run_async(ctx)) as agen:
        async for event in agen:
            yield event
```

### Running Multiple Sub-Agents Sequentially

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    # Stage 1: Run first agent
    async with Aclosing(self.agent1.run_async(ctx)) as agen:
        async for event in agen:
            yield event

    # Stage 2: Run second agent (can access state from agent1)
    async with Aclosing(self.agent2.run_async(ctx)) as agen:
        async for event in agen:
            yield event
```

### Conditional Sub-Agent Execution

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    # Run analyzer agent
    async with Aclosing(self.analyzer.run_async(ctx)) as agen:
        async for event in agen:
            yield event

    # Check result in state and decide next agent
    result = ctx.session.state.get("analysis_result")

    if result == "positive":
        # Run positive flow
        async with Aclosing(self.positive_handler.run_async(ctx)) as agen:
            async for event in agen:
                yield event
    else:
        # Run negative flow
        async with Aclosing(self.negative_handler.run_async(ctx)) as agen:
            async for event in agen:
                yield event
```

## Working with Session State

### Writing to State

Sub-agents (typically LlmAgent) write to state using the `output_key` parameter:

```python
# In LlmAgent initialization
story_generator = LlmAgent(
    name="story_generator",
    instruction="Generate a story about {topic}",
    output_key="current_story",  # Output saved to state["current_story"]
)
```

### Reading from State

Access state in instructions using placeholders:

```python
critic = LlmAgent(
    name="critic",
    instruction="Review this story: {current_story}",  # Reads from state
    output_key="criticism",
)
```

Or access directly in custom agent code:

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    current_story = ctx.session.state.get("current_story")
    topic = ctx.session.state.get("topic")

    # Your logic here
```

### Setting State Manually

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    # Set state values
    ctx.session.state["processed"] = True
    ctx.session.state["iteration_count"] = 0

    # Run sub-agent that can access these values
    async with Aclosing(self.sub_agent.run_async(ctx)) as agen:
        async for event in agen:
            yield event
```

## Complete Example: StoryFlowAgent

```python
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.utils.context_utils import Aclosing
from typing import AsyncGenerator


class StoryFlowAgent(BaseAgent):
    """Custom agent that orchestrates a multi-stage story generation workflow."""

    # Declare sub-agents as attributes
    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent

    def __init__(
        self,
        name: str,
        story_generator: LlmAgent,
        critic: LlmAgent,
        reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
    ):
        # Store sub-agents
        self.story_generator = story_generator
        self.critic = critic
        self.reviser = reviser
        self.grammar_check = grammar_check
        self.tone_check = tone_check

        # Create workflow agents
        from google.adk.agents.workflow.loop_agent import LoopAgent
        from google.adk.agents.workflow.sequential_agent import SequentialAgent

        # Loop for critic-reviser iteration
        loop_agent = LoopAgent(
            name="critic_reviser_loop",
            agents=[critic, reviser],
            max_iterations=3,
        )

        # Sequential check for grammar and tone
        sequential_agent = SequentialAgent(
            name="quality_checks",
            agents=[grammar_check, tone_check],
        )

        # Register top-level sub-agents with parent
        super().__init__(
            name=name,
            sub_agents=[story_generator, loop_agent, sequential_agent],
            description="Multi-stage story generation with quality checks",
        )

        # Store workflow agents for use in _run_async_impl
        self.loop_agent = loop_agent
        self.sequential_agent = sequential_agent

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Stage 1: Generate initial story
        async with Aclosing(self.story_generator.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # Stage 2: Run critic-reviser loop to improve story
        async with Aclosing(self.loop_agent.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # Stage 3: Run sequential quality checks
        async with Aclosing(self.sequential_agent.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # Stage 4: Conditional regeneration based on tone check
        tone_result = ctx.session.state.get("tone_check_result", "")

        if tone_result.lower() == "negative":
            # Tone is negative, regenerate the story
            async with Aclosing(self.story_generator.run_async(ctx)) as agen:
                async for event in agen:
                    yield event

        # Mark agent as complete
        ctx.set_agent_state(self.name, end_of_agent=True)
        yield self._create_agent_state_event(ctx)


# Define sub-agents with output_key to save results to state
story_generator = LlmAgent(
    name="story_generator",
    instruction="Generate a creative story about {topic}",
    output_key="current_story",
)

critic = LlmAgent(
    name="critic",
    instruction="Provide constructive criticism for this story: {current_story}",
    output_key="criticism",
)

reviser = LlmAgent(
    name="reviser",
    instruction=(
        "Revise the story based on this feedback: {criticism}\n"
        "Original story: {current_story}"
    ),
    output_key="current_story",  # Overwrites original story
)

grammar_check = LlmAgent(
    name="grammar_check",
    instruction="Check grammar in: {current_story}",
    output_key="grammar_suggestions",
)

tone_check = LlmAgent(
    name="tone_check",
    instruction=(
        "Analyze the tone of this story: {current_story}\n"
        "Respond with 'positive' or 'negative'"
    ),
    output_key="tone_check_result",
)

# Instantiate custom agent
story_flow_agent = StoryFlowAgent(
    name="StoryFlowAgent",
    story_generator=story_generator,
    critic=critic,
    reviser=reviser,
    grammar_check=grammar_check,
    tone_check=tone_check,
)
```

## Running the Custom Agent

```python
from google.adk.runners.runner import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

# Setup session
session_service = InMemorySessionService()
session = await session_service.create_session(
    app_name="StoryApp",
    user_id="user123",
    session_id="session456",
    state={"topic": "a brave knight"}  # Initial state
)

# Create runner
runner = Runner(
    agent=story_flow_agent,
    session_service=session_service,
)

# Execute agent
events = runner.run_async(
    user_id="user123",
    session_id="session456",
)

# Process events
async for event in events:
    if event.is_final_response():
        print(f"Final response: {event.content.parts[0].text}")

# Retrieve final state
final_session = await session_service.get_session(
    app_name="StoryApp",
    user_id="user123",
    session_id="session456",
)
print(f"Final state: {final_session.state}")
```

## Best Practices

### 1. Always Use Aclosing for Async Generators

```python
from google.adk.utils.context_utils import Aclosing

async with Aclosing(agent.run_async(ctx)) as agen:
    async for event in agen:
        yield event
```

### 2. Yield Events Immediately

Don't collect events in a list - yield them as they come for streaming:

```python
# GOOD - Streaming
async with Aclosing(agent.run_async(ctx)) as agen:
    async for event in agen:
        yield event

# BAD - Blocks streaming
events = []
async with Aclosing(agent.run_async(ctx)) as agen:
    async for event in agen:
        events.append(event)
for event in events:
    yield event
```

### 3. Use Descriptive State Keys

```python
# GOOD
ctx.session.state["current_story"] = "..."
ctx.session.state["tone_check_result"] = "positive"

# BAD
ctx.session.state["data"] = "..."
ctx.session.state["result"] = "positive"
```

### 4. Register All Top-Level Sub-Agents

```python
super().__init__(
    name=name,
    sub_agents=[agent1, loop_agent, sequential_agent],  # All top-level agents
)
```

### 5. Handle State Initialization

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    # Initialize state if needed
    if "iteration_count" not in ctx.session.state:
        ctx.session.state["iteration_count"] = 0

    # Your logic here
```

### 6. Mark Agent Completion for Resumability

```python
# At the end of _run_async_impl
if ctx.is_resumable:
    ctx.set_agent_state(self.name, end_of_agent=True)
    yield self._create_agent_state_event(ctx)
```

## When to Use Custom Agents

Use custom agents when you need:

- **Conditional logic** based on previous results
- **Complex multi-stage workflows** with dependencies
- **Dynamic agent selection** at runtime
- **Custom state management** beyond simple sequential/parallel flow
- **Integration with external systems** or APIs
- **Patterns beyond** Sequential, Parallel, or Loop agents

For simpler workflows, prefer:
- `LlmAgent` for single LLM interactions
- `SequentialAgent` for linear workflows
- `ParallelAgent` for concurrent execution
- `LoopAgent` for iterative refinement

## Advanced: Handling Transfers and Resumption

### Checking for Sub-Agent Resume

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    agent_state = self._load_agent_state(ctx, BaseAgentState)

    # Check if we need to resume a sub-agent
    if agent_state is not None and (
        agent_to_transfer := self._get_subagent_to_resume(ctx)
    ):
        async with Aclosing(agent_to_transfer.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        ctx.set_agent_state(self.name, end_of_agent=True)
        yield self._create_agent_state_event(ctx)
        return

    # Normal execution continues...
```

### Accessing Event History

```python
# Get all events from current invocation and branch
events = ctx._get_events(current_invocation=True, current_branch=True)

# Check last event
if events:
    last_event = events[-1]
    print(f"Last event author: {last_event.author}")
```

## Debugging Tips

1. **Log state changes**: Print state at key points
2. **Trace event flow**: Log when yielding events
3. **Check state keys**: Verify keys exist before accessing
4. **Use breakpoints**: Debug async generators carefully
5. **Test each stage**: Verify sub-agents work independently first

## Summary

Custom agents provide maximum flexibility for complex orchestration:

1. Extend `BaseAgent`
2. Implement `_run_async_impl` (and/or `_run_live_impl`)
3. Use `ctx.session.state` for inter-agent communication
4. Invoke sub-agents with `agent.run_async(ctx)`
5. Yield events immediately for streaming
6. Use `Aclosing` for proper cleanup
7. Mark completion with state events
