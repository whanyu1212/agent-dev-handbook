# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational handbook for building AI agents using modern frameworks. The repository contains reference implementations, examples, and patterns for developing agentic systems, with primary focus on:

- **LangGraph**: Building stateful, graph-based agents with cyclic workflows
- **Google ADK**: Leveraging Google's Agent Development Kit for scalable agent architectures

The codebase is structured as a Python package (`agent-dev-handbook`) providing educational examples and notes rather than a production library.

## Development Commands

### Environment Setup
This project uses `uv` for dependency management:

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Install with dev dependencies
uv sync --all-extras
```

### Code Quality

```bash
# Run linter (with auto-fix)
ruff check --fix .

# Run formatter
ruff format .

# Run type checker (manual stage)
mypy --ignore-missing-imports --no-strict-optional src/

# Run pre-commit hooks on all files
pre-commit run --all-files

# Run pre-commit with mypy (manual stage)
pre-commit run --hook-stage manual mypy
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_specific.py

# Run with verbose output
pytest -v

# Run async tests (configured to auto-detect)
pytest test/test_async.py
```

Note: Test files are located in `test/` directory and follow naming conventions: `test_*.py` or `*_test.py`.

## Architecture

### Directory Structure

```
src/agent_dev_handbook/
├── langgraph/          # LangGraph implementations
│   ├── basics/         # Fundamental concepts
│   ├── tools/          # Tool integration patterns
│   ├── memory/         # Memory management
│   ├── patterns/       # Common agent patterns
│   └── examples/       # Complete examples
├── adk/                # Google ADK implementations
│   ├── basics/         # Core ADK concepts
│   ├── tools/          # Tool definitions
│   ├── mcp/            # MCP integration
│   ├── examples/       # Working examples
│   └── notes/          # Implementation guides (markdown)
└── shared/             # Cross-framework utilities
    ├── evaluations/    # Agent evaluation tools
    ├── utils/          # Common utilities
    └── prompts/        # Shared prompts
```

### Key ADK Concepts

#### Event-Driven Architecture

ADK agents communicate through **Events**, not return values. Events serve as:
- Immutable conversation history (stored in `session.events`)
- Streaming protocol for real-time UI updates
- Coordination mechanism between agents
- Debugging trace and audit trail

**Session State** (`session.state`) is a mutable key-value store for passing data between agents, separate from the immutable event history.

#### BaseAgent Implementation

All ADK agents extend `BaseAgent` and must implement:

```python
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    """Core logic for text-based conversation."""
    # Must yield at least one Event
    yield Event(...)
```

**Key patterns:**
- Use `Aclosing` when invoking sub-agents: `async with Aclosing(agent.run_async(ctx)) as agen`
- Yield events immediately for streaming (don't collect and batch)
- Access state via `ctx.session.state` for inter-agent communication
- Mark completion with `ctx.set_agent_state(self.name, end_of_agent=True)` for resumability
- Use `output_key` in `LlmAgent` to save outputs to session state

#### Custom Agents vs Workflow Agents

**Use built-in workflow agents when possible:**
- `LlmAgent` - Single LLM interaction with optional tools
- `SequentialAgent` - Linear execution of sub-agents
- `ParallelAgent` - Concurrent execution
- `LoopAgent` - Iterative refinement

**Create custom agents (extending `BaseAgent`) when you need:**
- Conditional logic based on previous results
- Complex multi-stage workflows with dependencies
- Dynamic agent selection at runtime
- Custom state management patterns
- Integration with external systems

#### DummyAgent Pattern

`DummyAgent` (see `src/agent_dev_handbook/adk/examples/dummy_agent.py`) implements deterministic nodes similar to LangGraph nodes:
- Execute predefined logic without LLM calls
- Useful for data validation, routing, preprocessing
- Can execute tools programmatically
- Fully integrates with ADK's event system

### Important Implementation Details

#### Agent Hierarchy
- Agent names must be valid Python identifiers and unique within the agent tree
- Sub-agents are registered via `sub_agents` parameter in `super().__init__()`
- Parent agent is automatically set when sub-agents are added
- Use `clone()` to reuse an agent in multiple places

#### State Management
- LLM agents write to state using `output_key` parameter
- Access state in instructions using placeholders: `"{variable_name}"`
- Custom agents access state directly: `ctx.session.state["key"]`
- State persists across agent invocations within a session

#### Event Yielding
```python
# GOOD - Streaming
async with Aclosing(agent.run_async(ctx)) as agen:
    async for event in agen:
        yield event  # Immediate streaming

# BAD - Blocks streaming
events = []
async with Aclosing(agent.run_async(ctx)) as agen:
    async for event in agen:
        events.append(event)
for event in events:
    yield event
```

## Code Style

- Line length: 88 characters (Black-compatible)
- Python version: 3.12+
- Linting: Ruff with flake8 (E/F) and isort (I) rules
- Type hints: Use mypy for type checking (with lenient settings for examples)
- Imports: Sorted automatically by ruff/isort

## Educational Notes

The `src/agent_dev_handbook/adk/notes/` directory contains detailed markdown guides:
- `base_agent_notes.md` - BaseAgent class reference
- `custom_agent_implementation_guide.md` - Step-by-step custom agent patterns
- `why_events_in_adk.md` - Deep dive into event-driven architecture
- `dummy_agent_guide.md` - Deterministic node patterns
- `dynamic_tools_and_prompts.md` - Runtime tool/prompt configuration
- `performance_considerations.md` - Performance optimization tips
- `callbacks_reference.md` - Callback patterns and debugging hooks
- `exploring_adk_internals.md` - Internal implementation details

These notes are essential reading when implementing new agent patterns.

## ADK Examples

The `src/agent_dev_handbook/adk/examples/` directory contains runnable examples:
- `dummy_agent.py` - DummyAgent implementation for deterministic logic nodes
- `custom_agent.py` - ConditionalStoryAgent demonstrating custom orchestration
- `run_dummy_agent.py` - Full 3-agent workflow with state flow visualization
- `callback_inspector.py` - Debugging utility for inspecting model callbacks
- `run_callback_inspector.py` - Example using the callback inspector
- `test_dynamic_multiturn.py` - Multi-turn conversation with dynamic prompts/tools

Run examples directly: `python src/agent_dev_handbook/adk/examples/run_dummy_agent.py`

## Common Pitfalls

1. **Forgetting to yield events** - Every agent method must yield at least one Event
2. **Not using Aclosing** - Always wrap sub-agent calls with `async with Aclosing(...)`
3. **Blocking streaming** - Yield events immediately, don't collect first
4. **Missing agent completion** - Mark agents complete with `ctx.set_agent_state()` for resumability
5. **Incorrect state keys** - Ensure `output_key` matches the placeholder used in downstream agents
6. **Not registering sub-agents** - All top-level sub-agents must be in the `sub_agents` list
