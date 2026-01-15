# BaseAgent Class - Google ADK

## Overview
`BaseAgent` is the base class for all agents in the Google Agent Development Kit (ADK). It extends Pydantic's `BaseModel` and provides the core functionality for building custom agents.

## Key Attributes

### Required Attributes
- **`name: str`** - Agent's name (must be a valid Python identifier, unique within agent tree, cannot be "user")
- **`description: str = ''`** - One-line description of agent's capability (used by the model to determine delegation)

### Agent Hierarchy
- **`parent_agent: Optional[BaseAgent]`** - The parent agent (read-only, set automatically)
- **`sub_agents: list[BaseAgent]`** - List of sub-agents that this agent can delegate to
- **`root_agent: BaseAgent`** (property) - Gets the root agent of the agent tree

### Callbacks
- **`before_agent_callback: Optional[BeforeAgentCallback]`** - Called before agent runs
  - Can skip agent execution by returning content
  - Can be a single callback or list of callbacks
- **`after_agent_callback: Optional[AfterAgentCallback]`** - Called after agent runs
  - Can append additional response content
  - Can be a single callback or list of callbacks

### Configuration
- **`config_type: ClassVar[type[BaseAgentConfig]]`** - The config type for the agent
  - Override in subclasses to use custom config types

## Core Methods to Implement

### Required Implementations
When creating a custom agent, you **must** implement ONE OR BOTH of these methods:

1. **`async _run_async_impl(ctx: InvocationContext) -> AsyncGenerator[Event, None]`**
   - Core logic for text-based conversation
   - Yields events as the agent processes requests
   - Raises `NotImplementedError` by default

2. **`async _run_live_impl(ctx: InvocationContext) -> AsyncGenerator[Event, None]`**
   - Core logic for video/audio-based conversation
   - Yields events during live interactions
   - Raises `NotImplementedError` by default

### Public Entry Points (Do NOT Override)
These are marked `@final` and should not be overridden:

- **`async run_async(parent_context: InvocationContext) -> AsyncGenerator[Event, None]`**
  - Entry method to run agent via text-based conversation
  - Handles callbacks, tracing, and invocation management

- **`async run_live(parent_context: InvocationContext) -> AsyncGenerator[Event, None]`**
  - Entry method to run agent via video/audio-based conversation
  - Handles callbacks, tracing, and invocation management

## Utility Methods

### Agent Discovery
- **`find_agent(name: str) -> Optional[BaseAgent]`**
  - Finds agent with given name in this agent and descendants

- **`find_sub_agent(name: str) -> Optional[BaseAgent]`**
  - Finds agent with given name in descendants only

### Agent Cloning
- **`clone(update: Mapping[str, Any] | None = None) -> BaseAgent`**
  - Creates a deep copy of the agent
  - Can update specific fields via the `update` parameter
  - Recursively clones sub-agents
  - Cannot update `parent_agent` field

### Configuration
- **`@classmethod from_config(config: BaseAgentConfig, config_abs_path: str) -> BaseAgent`** (experimental)
  - Creates an agent from a config object
  - Override `_parse_config()` to customize config parsing

- **`@classmethod _parse_config(config: BaseAgentConfig, config_abs_path: str, kwargs: Dict[str, Any]) -> Dict[str, Any]`**
  - Hook for subclasses to customize config parsing
  - Returns updated kwargs for agent constructor

## State Management

### Internal Methods
- **`_load_agent_state(ctx: InvocationContext, state_type: Type[AgentState]) -> Optional[AgentState]`**
  - Loads agent state from invocation context
  - Returns None if state doesn't exist

- **`_create_agent_state_event(ctx: InvocationContext) -> Event`**
  - Creates an event with current agent state
  - Used to persist state changes

- **`_create_invocation_context(parent_context: InvocationContext) -> InvocationContext`**
  - Creates new invocation context for this agent

## Validation

### Field Validators
- **`validate_name(value: str)`** - Ensures name is a valid Python identifier and not "user"
- **`validate_sub_agents_unique_names(value: list[BaseAgent])`** - Warns about duplicate sub-agent names

### Model Configuration
```python
model_config = ConfigDict(
    arbitrary_types_allowed=True,
    extra='forbid',  # Prevents extra fields
)
```

## Important Notes

1. **Agent Names**:
   - Must be valid Python identifiers (start with letter/underscore, contain only letters/digits/underscores)
   - Must be unique within the agent tree
   - Cannot be "user" (reserved for end-user input)

2. **Sub-Agent Management**:
   - An agent can only be added as a sub-agent once
   - To reuse an agent, create multiple instances with `clone()`
   - Parent agent is automatically set when sub-agents are added

3. **Callbacks**:
   - Callbacks can be single functions or lists of functions
   - Before callbacks can short-circuit agent execution
   - After callbacks can append additional content
   - Plugin callbacks run before canonical callbacks

4. **Inheritance**:
   - Extends Pydantic's `BaseModel`
   - Inherits all Pydantic validation and serialization features
   - Use `model_copy()`, `model_dump()`, `model_validate()` for Pydantic operations

## Example Usage Pattern

```python
from google.adk.agents import BaseAgent
from google.adk.types import Event, InvocationContext

class CustomAgent(BaseAgent):
    # Optional: Custom config type
    # config_type: ClassVar[type[BaseAgentConfig]] = CustomAgentConfig

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Your agent logic here
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content="Agent response",
        )
```

## Pydantic Features Available

Since BaseAgent extends BaseModel, you have access to:
- `model_dump()` - Convert to dict
- `model_dump_json()` - Convert to JSON string
- `model_validate()` - Validate and create from dict
- `model_validate_json()` - Validate and create from JSON
- `model_copy()` - Create a copy with optional updates
- `model_fields` - Get all field definitions
- `model_config` - Access model configuration
