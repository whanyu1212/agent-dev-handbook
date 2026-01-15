# AGENTS.md

This file provides guidance to agents when working with code in this repository.

# Project Context
- **Stack**: Python 3.12+, `uv` for dependency management.
- **Package**: `agent-dev-handbook` (educational examples, not production library).
- **No pytest Tests**: `pyproject.toml` references a `test` directory, but it **DOES NOT EXIST** yet.
- **"Tests" are Scripts**: Files like `test_dynamic_multiturn.py` in examples are runnable scripts (`python path/to/file.py`), NOT pytest unit tests.

# Commands
- **Install**: `uv sync` (or `uv sync --all-extras` for dev tools).
- **Lint/Format**: `ruff check --fix .` and `ruff format .`.
- **Type Check**: `mypy --ignore-missing-imports --no-strict-optional src/`.
- **Run Examples**: Execute directly with python, e.g., `python src/agent_dev_handbook/adk/examples/run_dummy_agent.py`.

# Linting Rules (IMPORTANT for code generation)
**Ruff config** (`select = ["E", "F", "I"]`):
- **Line length**: 88 chars max (E501) - split long strings, extract variables
- **Unused imports** (F401): Remove them
- **Unused variables** (F841): Remove or prefix with `_`
- **Import order** (I001): stdlib → third-party → local, alphabetized

**Mypy**: Uses `--ignore-missing-imports --no-strict-optional` (lenient for examples)

# ADK Examples (src/agent_dev_handbook/adk/examples/)
- **`dummy_agent.py`** - DummyAgent for deterministic logic without LLM calls (validation, routing, preprocessing).
- **`custom_agent.py`** - ConditionalStoryAgent showing custom orchestration with conditional logic.
- **`run_dummy_agent.py`** - Full 3-agent sequential workflow with rich console output.
- **`callback_inspector.py`** - Debugging utility: inspects before/after model callbacks, tracks state/tools/usage.
- **`run_callback_inspector.py`** - Example demonstrating callback inspector usage.
- **`test_dynamic_multiturn.py`** - Multi-turn demo with dynamic prompt/tool changes (tier upgrades).

# ADK Notes (src/agent_dev_handbook/adk/notes/)
Essential markdown guides for understanding ADK patterns:
- `why_events_in_adk.md` - Event-driven architecture deep dive
- `base_agent_notes.md` - BaseAgent class reference
- `custom_agent_implementation_guide.md` - Custom agent patterns
- `dummy_agent_guide.md` - Deterministic node implementation
- `dynamic_tools_and_prompts.md` - Runtime configuration
- `callbacks_reference.md` - Callback patterns and hooks
- `exploring_adk_internals.md` - Internal implementation details
- `performance_considerations.md` - Optimization tips

# Code Patterns (Non-Obvious)
- **ADK Events**: Agents communicate via `Event` streams, not return values. MUST yield `Event` objects.
- **Streaming Safety**: MUST use `async with Aclosing(agent.run_async(ctx)) as agen:` when calling sub-agents to support streaming.
- **Agent Registration**: Sub-agents MUST be registered in `super().__init__(..., sub_agents=[...])` for hierarchy and state management to work.
- **State Access**: Use `ctx.session.state` (mutable) for inter-agent data, `session.events` (immutable) for history.
- **Completion Signal**: Resume capability requires explicit `ctx.set_agent_state(self.name, end_of_agent=True)`.
- **Output Keys**: Use `output_key` parameter in `LlmAgent` to save outputs to state; access in downstream agents with `"{key_name}"` placeholders.

# Gotchas
- **Empty Directories**: `langgraph/`, `shared/`, and some `adk/` subdirectories (basics, tools, mcp) are placeholder structure only.
- **Import Cycles**: Be aware of potential circular imports in monorepo-style structure; explicit types package usage prevents this.
- **Callback Inspector Config**: Use `InspectorConfig` to control verbosity; default logs everything which can be noisy.
