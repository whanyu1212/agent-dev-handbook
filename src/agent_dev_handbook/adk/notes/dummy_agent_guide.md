# Creating Dummy Agents in ADK

## What is a Dummy Agent?

A **Dummy Agent** is a deterministic agent that executes pure Python logic **without making LLM calls**. It's similar to a "node" in LangGraph - it performs programmatic operations while fully integrating with ADK's event system, state management, and agent orchestration.

## Why Use Dummy Agents?

Use dummy agents for:

- ✅ **Data validation and transformation** - Parse, validate, or format data
- ✅ **Deterministic routing** - Route based on state values without LLM reasoning
- ✅ **Tool orchestration** - Call tools programmatically without LLM involvement
- ✅ **Pre/post-processing steps** - Clean data before/after LLM calls
- ✅ **Mathematical operations** - Calculations, aggregations, statistics
- ✅ **State mutations** - Update state based on business logic
- ✅ **API calls** - Fetch external data deterministically

**Don't use dummy agents** when you need LLM reasoning, natural language understanding, or open-ended responses.

---

## Basic Implementation

### Minimal Dummy Agent

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from typing import AsyncGenerator, Callable, Optional


class DummyAgent(BaseAgent):
    """Agent that executes deterministic logic without LLM calls."""

    logic_function: Callable[[InvocationContext], Any] = lambda ctx: None
    output_key: Optional[str] = None

    def __init__(
        self,
        name: str,
        description: str = "",
        logic_function: Optional[Callable[[InvocationContext], Any]] = None,
        output_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            logic_function=logic_function or (lambda ctx: None),
            output_key=output_key,
            sub_agents=[],
            **kwargs,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Execute deterministic logic and emit events."""

        # 1. Execute the logic function
        result = self.logic_function(ctx)

        # Handle async functions
        if hasattr(result, '__await__'):
            result = await result

        # 2. Save to state if output_key is set
        if self.output_key and result is not None:
            ctx.session.state[self.output_key] = result

        # 3. Emit event (goes into conversation history)
        if result:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role='model',
                    parts=[types.Part(text=str(result))]
                ),
            )

        # 4. Mark completion
        if ctx.is_resumable:
            ctx.set_agent_state(self.name, end_of_agent=True)
            yield self._create_agent_state_event(ctx)
```

---

## Usage Examples

### Example 1: Data Validation

```python
def validate_email(ctx: InvocationContext):
    """Validate email format."""
    email = ctx.session.state.get("user_email", "")

    if not email or "@" not in email:
        return {
            "valid": False,
            "error": "Invalid email format"
        }

    return {
        "valid": True,
        "message": f"Email {email} is valid"
    }

validator = DummyAgent(
    name="email_validator",
    description="Validates email addresses",
    logic_function=validate_email,
    output_key="validation_result",
)

# Use in workflow
from google.adk.agents import LlmAgent, SequentialAgent

workflow = SequentialAgent(
    name="signup_flow",
    sub_agents=[
        validator,                    # Validate email first
        LlmAgent(                     # Then generate welcome message
            name="welcomer",
            instruction="Generate welcome email for {user_email}",
        ),
    ]
)
```

### Example 2: Conditional Routing

```python
def route_by_priority(ctx: InvocationContext):
    """Route based on priority level."""
    priority = ctx.session.state.get("priority", "medium").lower()

    routing_map = {
        "high": "escalation_agent",
        "medium": "standard_agent",
        "low": "automated_agent",
    }

    target = routing_map.get(priority, "standard_agent")
    ctx.session.state["routed_to"] = target

    return f"Routing {priority} priority to {target}"

router = DummyAgent(
    name="priority_router",
    description="Routes requests by priority",
    logic_function=route_by_priority,
    output_key="routing_decision",
)
```

### Example 3: Data Transformation

```python
def transform_user_data(ctx: InvocationContext):
    """Transform raw user data into structured format."""
    raw = ctx.session.state.get("raw_user_data", {})

    transformed = {
        "user_id": raw.get("id"),
        "full_name": f"{raw.get('first_name', '')} {raw.get('last_name', '')}".strip(),
        "email_domain": raw.get("email", "").split("@")[-1] if "@" in raw.get("email", "") else "",
        "is_premium": raw.get("tier") == "premium",
        "created_at": datetime.now().isoformat(),
    }

    return transformed

transformer = DummyAgent(
    name="data_transformer",
    description="Transforms raw user data",
    logic_function=transform_user_data,
    output_key="user_profile",
)
```

### Example 4: Mathematical Operations

```python
def calculate_statistics(ctx: InvocationContext):
    """Calculate statistics from data."""
    numbers = ctx.session.state.get("numbers", [])

    if not numbers:
        return "No data to analyze"

    stats = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }

    ctx.session.state["statistics"] = stats

    return f"Analyzed {stats['count']} numbers: mean={stats['mean']:.2f}, range=[{stats['min']}, {stats['max']}]"

calculator = DummyAgent(
    name="stat_calculator",
    description="Calculates statistics",
    logic_function=calculate_statistics,
    output_key="stats_summary",
)
```

### Example 5: Async Operations

```python
import aiohttp

async def fetch_external_data(ctx: InvocationContext):
    """Fetch data from external API."""
    user_id = ctx.session.state.get("user_id")

    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as resp:
            data = await resp.json()

    ctx.session.state["external_data"] = data
    return f"Fetched data for user {user_id}"

fetcher = DummyAgent(
    name="api_fetcher",
    description="Fetches external API data",
    logic_function=fetch_external_data,
    output_key="fetch_result",
)
```

---

## Complete Workflow Example

Here's a full example showing dummy agents in a workflow:

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# 1. LLM Agent - Generates a number
generator = LlmAgent(
    name="generator",
    instruction="Generate a random number between 1 and 100. Just respond with the number.",
    output_key="generated_number",
)

# 2. Dummy Agent - Doubles the number (NO LLM)
def double_number(ctx):
    """Pure Python logic - no LLM involved."""
    num_str = ctx.session.state.get("generated_number", "0")

    # Extract number from LLM response
    try:
        num = int(''.join(filter(str.isdigit, str(num_str))))
    except:
        num = 0

    doubled = num * 2

    # Write to state for next agent
    ctx.session.state["doubled_number"] = doubled

    # Return message that goes into event
    return f"I doubled {num} to get {doubled}"

doubler = DummyAgent(
    name="doubler",
    description="Doubles a number without using LLM",
    logic_function=double_number,
    output_key="doubler_result",
)

# 3. LLM Agent - Interprets the result
interpreter = LlmAgent(
    name="interpreter",
    instruction=(
        "The original number was {generated_number}. "
        "It was doubled to {doubled_number}. "
        "Explain this calculation in a friendly way."
    ),
    output_key="final_response",
)

# Create workflow
workflow = SequentialAgent(
    name="number_workflow",
    sub_agents=[generator, doubler, interpreter],
)

# Setup and run
async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="DummyTest",
        user_id="user1",
        session_id="session1",
        state={},
    )

    runner = Runner(
        app_name="DummyTest",
        agent=workflow,
        session_service=session_service
    )

    events = runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=types.Content(
            role='user',
            parts=[types.Part(text="Start")]
        )
    )

    async for event in events:
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                print(f"[{event.author}]: {text}")

    # Check final state
    session = await session_service.get_session(
        app_name="DummyTest",
        user_id="user1",
        session_id="session1"
    )
    print(f"\nFinal state: {session.state}")

# Run
import asyncio
asyncio.run(main())
```

**Output:**
```
[generator]: 42
[doubler]: I doubled 42 to get 84
[interpreter]: Great! Let me explain: We started with 42, and when we double...

Final state: {
    'generated_number': '42',
    'doubled_number': 84,
    'doubler_result': 'I doubled 42 to get 84',
    'final_response': 'Great! Let me explain...'
}
```

---

## Key Concepts

### 1. Events vs State

**Events** (conversation history):
- User sees these
- Immutable record of what happened
- Dummy agents emit events via `yield Event(...)`

**State** (data passing):
- User doesn't see these
- Mutable workspace for agents
- Dummy agents write via `ctx.session.state[key] = value`

```python
def my_logic(ctx):
    # Read from state
    input_data = ctx.session.state.get("input")

    # Do something
    result = process(input_data)

    # Write to state (for next agent)
    ctx.session.state["processed_data"] = result

    # Return message (goes to events)
    return f"Processed {len(result)} items"
```

### 2. Reading State from Previous Agents

```python
def use_previous_results(ctx):
    """Access results from previous agents."""

    # Read from LlmAgent output
    llm_response = ctx.session.state.get("llm_output_key")

    # Read from another DummyAgent
    validation = ctx.session.state.get("validation_result")

    # Use the data
    if validation.get("valid"):
        return f"Processing: {llm_response}"
    else:
        return f"Error: {validation.get('error')}"
```

### 3. Sync vs Async Logic Functions

```python
# Sync function (simpler)
def sync_logic(ctx):
    result = do_something()
    return result

# Async function (for I/O operations)
async def async_logic(ctx):
    result = await async_operation()
    return result

# Both work with DummyAgent!
agent1 = DummyAgent(name="sync", logic_function=sync_logic)
agent2 = DummyAgent(name="async", logic_function=async_logic)
```

### 4. No Return Value (Silent Agents)

```python
def silent_state_updater(ctx):
    """Update state without generating visible output."""
    ctx.session.state["processed"] = True
    ctx.session.state["timestamp"] = datetime.now().isoformat()
    # Return None - no event emitted, just state updated
    return None

silent_agent = DummyAgent(
    name="silent_updater",
    logic_function=silent_state_updater,
)
```

---

## Best Practices

### ✅ DO:

1. **Use for deterministic operations**
   ```python
   # Good - predictable logic
   def validate_age(ctx):
       age = ctx.session.state.get("age", 0)
       return age >= 18
   ```

2. **Write clear state keys**
   ```python
   # Good - descriptive keys
   ctx.session.state["email_validated"] = True
   ctx.session.state["user_tier"] = "premium"

   # Bad - unclear keys
   ctx.session.state["flag"] = True
   ctx.session.state["data"] = "premium"
   ```

3. **Handle missing state gracefully**
   ```python
   # Good - defensive programming
   data = ctx.session.state.get("input_data", [])
   if not data:
       return "No data available"
   ```

4. **Use output_key for important results**
   ```python
   DummyAgent(
       name="validator",
       logic_function=validate,
       output_key="validation_result",  # Save for later use
   )
   ```

### ❌ DON'T:

1. **Don't use for tasks requiring LLM reasoning**
   ```python
   # Bad - use LlmAgent instead
   def generate_creative_story(ctx):
       return "Once upon a time..."  # This needs LLM!
   ```

2. **Don't make external calls without error handling**
   ```python
   # Bad - no error handling
   def fetch_data(ctx):
       return requests.get(url).json()  # What if it fails?

   # Good - handle errors
   def fetch_data(ctx):
       try:
           return requests.get(url, timeout=5).json()
       except Exception as e:
           ctx.session.state["fetch_error"] = str(e)
           return {"error": str(e)}
   ```

3. **Don't forget to emit events for important operations**
   ```python
   # Bad - no feedback
   def process(ctx):
       ctx.session.state["result"] = compute()
       # User sees nothing!

   # Good - return status
   def process(ctx):
       result = compute()
       ctx.session.state["result"] = result
       return f"Processed: {result}"  # User sees this
   ```

---

## Comparison: Dummy Agent vs LLM Agent

| Feature | Dummy Agent | LLM Agent |
|---------|------------|-----------|
| **Makes LLM calls** | ❌ No | ✅ Yes |
| **Cost** | Free (just compute) | Costs per token |
| **Speed** | Fast (milliseconds) | Slower (seconds) |
| **Deterministic** | ✅ Always same output | ❌ Varies each time |
| **Natural language** | ❌ Not needed | ✅ Understands NL |
| **Reasoning** | ❌ Only programmed logic | ✅ Can reason |
| **Use for** | Validation, routing, math, data transform | Generation, understanding, reasoning |

---

## Common Patterns

### Pattern 1: Validate → Process → Report

```python
validator = DummyAgent(name="validate", logic_function=validate_input)
processor = LlmAgent(name="process", instruction="Process {input}")
reporter = DummyAgent(name="report", logic_function=generate_report)

workflow = SequentialAgent(sub_agents=[validator, processor, reporter])
```

### Pattern 2: Fetch → Analyze → Act

```python
fetcher = DummyAgent(name="fetch", logic_function=fetch_data)
analyzer = LlmAgent(name="analyze", instruction="Analyze: {data}")
actor = DummyAgent(name="act", logic_function=take_action)

workflow = SequentialAgent(sub_agents=[fetcher, analyzer, actor])
```

### Pattern 3: Route Based on State

```python
from google.adk.agents import BaseAgent

class ConditionalRouter(BaseAgent):
    async def _run_async_impl(self, ctx):
        # Use dummy agent for routing decision
        router = DummyAgent(name="router", logic_function=route_logic)
        async with Aclosing(router.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # Route based on decision
        target = ctx.session.state.get("routed_to")
        target_agent = self.find_sub_agent(target)

        async with Aclosing(target_agent.run_async(ctx)) as agen:
            async for event in agen:
                yield event
```

---

## Debugging Tips

1. **Print inside logic functions**
   ```python
   def debug_logic(ctx):
       print(f"State: {ctx.session.state}")  # See what's available
       result = process()
       print(f"Result: {result}")  # See what you're returning
       return result
   ```

2. **Check state after each agent**
   ```python
   session = await session_service.get_session(...)
   print(f"State after validator: {session.state}")
   ```

3. **Test logic functions independently**
   ```python
   # Create mock context
   class MockContext:
       def __init__(self):
           self.session = type('obj', (object,), {'state': {}})()

   ctx = MockContext()
   ctx.session.state["input"] = "test"
   result = my_logic_function(ctx)
   print(result)
   ```

---

## Summary

**Dummy Agents** provide a way to inject deterministic, programmatic logic into your agent workflows without the cost, latency, or non-determinism of LLM calls. They're perfect for:

- Data validation and transformation
- Conditional routing
- Mathematical operations
- API calls and external integrations
- Pre/post-processing steps

Use them as "glue" between LLM agents to create sophisticated, efficient workflows that combine the best of deterministic programming and AI reasoning.

**Remember**: If it can be written as a Python function, it can be a Dummy Agent!
