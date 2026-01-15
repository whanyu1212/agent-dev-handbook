# Dynamic Tools and Prompts in ADK

## Overview

Yes! ADK supports **dynamic tool loading** and **dynamic prompt generation** at runtime. You can modify both tools and instructions based on:
- Session state
- User permissions
- Runtime conditions
- Previous agent outputs
- External data

## Two Approaches

### 1. **InstructionProvider** - Dynamic Prompts
Use a function instead of a static string for `instruction`

### 2. **before_model_callback** - Dynamic Tools & Prompts
Intercept and modify the LLM request before it's sent

---

## Approach 1: Dynamic Prompts with InstructionProvider

### Basic Pattern

Instead of a static string, provide a function that generates the instruction dynamically:

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

def dynamic_instruction(ctx: ReadonlyContext) -> str:
    """Generate instruction based on session state."""
    user_tier = ctx.session.state.get("user_tier", "free")

    if user_tier == "premium":
        return "You are a premium support agent. Provide detailed, expert assistance."
    else:
        return "You are a basic support agent. Provide standard assistance."

agent = LlmAgent(
    name="support_agent",
    instruction=dynamic_instruction,  # Function, not string!
)
```

### Async InstructionProvider

You can also use async functions to fetch data:

```python
async def async_instruction(ctx: ReadonlyContext) -> str:
    """Fetch instruction from database or API."""
    user_id = ctx.session.state.get("user_id")

    # Fetch from external source
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/instructions/{user_id}") as resp:
            custom_instruction = await resp.text()

    return custom_instruction

agent = LlmAgent(
    name="custom_agent",
    instruction=async_instruction,
)
```

### Examples

#### Example 1: Instruction Based on User Role

```python
def role_based_instruction(ctx: ReadonlyContext) -> str:
    """Change behavior based on user role."""
    role = ctx.session.state.get("user_role", "guest")

    instructions = {
        "admin": "You are an admin assistant with full access. Be concise and technical.",
        "user": "You are a friendly assistant. Be helpful and explain clearly.",
        "guest": "You are a limited assistant. Only provide basic information.",
    }

    base = instructions.get(role, instructions["guest"])

    # Add context from state
    username = ctx.session.state.get("username", "User")
    return f"{base}\n\nYou are helping {username}."

agent = LlmAgent(
    name="adaptive_agent",
    instruction=role_based_instruction,
)
```

#### Example 2: Instruction Based on Conversation History

```python
def context_aware_instruction(ctx: ReadonlyContext) -> str:
    """Adapt instruction based on conversation progress."""
    turn_count = ctx.session.state.get("turn_count", 0)

    if turn_count == 0:
        return "You are starting a new conversation. Be welcoming and ask clarifying questions."
    elif turn_count < 3:
        return "Continue the conversation. Build on previous context and gather information."
    else:
        return "You've had several exchanges. Provide comprehensive answers and summarize if needed."

agent = LlmAgent(
    name="conversational_agent",
    instruction=context_aware_instruction,
)
```

#### Example 3: Instruction with Template Variables

```python
def templated_instruction(ctx: ReadonlyContext) -> str:
    """Build instruction with dynamic data."""
    # Get data from state
    language = ctx.session.state.get("language", "English")
    tone = ctx.session.state.get("tone", "professional")
    domain = ctx.session.state.get("domain", "general")

    instruction = f"""You are a {domain} expert assistant.

Language: Respond in {language}
Tone: Use a {tone} tone
Context: {ctx.session.state.get('user_context', 'No additional context')}

Provide accurate, helpful responses tailored to these requirements."""

    return instruction

agent = LlmAgent(
    name="templated_agent",
    instruction=templated_instruction,
)
```

---

## Approach 2: Dynamic Tools with before_model_callback

The `before_model_callback` gives you full control to modify the LLM request before it's sent.

### Basic Pattern

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool

def dynamic_tools_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Dynamically add/remove tools based on context."""

    # Access session state
    user_tier = callback_context.session.state.get("user_tier", "free")

    # Modify tools based on tier
    if user_tier == "premium":
        # Add premium tools
        def premium_feature():
            """Premium-only feature."""
            return "Premium feature result"

        llm_request.tools_dict["premium_feature"] = FunctionTool(func=premium_feature)
    else:
        # Remove premium tools if they exist
        if "premium_feature" in llm_request.tools_dict:
            del llm_request.tools_dict["premium_feature"]

    # Return None to continue with modified request
    return None

agent = LlmAgent(
    name="tiered_agent",
    instruction="Help the user with available tools",
    tools=[],  # Start with base tools
    before_model_callback=dynamic_tools_callback,
)
```

### What You Can Modify in before_model_callback

```python
def comprehensive_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Shows all modifiable aspects of LlmRequest."""

    # 1. Modify tools
    llm_request.tools_dict["new_tool"] = FunctionTool(func=my_func)
    del llm_request.tools_dict["unwanted_tool"]

    # 2. Modify contents (conversation history)
    llm_request.contents.append(
        types.Content(
            role='user',
            parts=[types.Part(text="Additional context")]
        )
    )

    # 3. Modify config (temperature, max_tokens, etc.)
    llm_request.config.temperature = 0.7
    llm_request.config.max_output_tokens = 1000

    # 4. Append instructions
    llm_request.append_instructions(["Additional instruction"])

    return None  # Continue with modified request
```

### Examples

#### Example 1: Permission-Based Tool Loading

```python
def permission_based_tools(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Load tools based on user permissions."""
    permissions = callback_context.session.state.get("permissions", [])

    # Define tool sets
    admin_tools = {
        "delete_user": FunctionTool(func=delete_user),
        "modify_settings": FunctionTool(func=modify_settings),
    }

    power_user_tools = {
        "advanced_search": FunctionTool(func=advanced_search),
        "export_data": FunctionTool(func=export_data),
    }

    basic_tools = {
        "search": FunctionTool(func=basic_search),
        "view_info": FunctionTool(func=view_info),
    }

    # Add tools based on permissions
    if "admin" in permissions:
        llm_request.tools_dict.update(admin_tools)
        llm_request.tools_dict.update(power_user_tools)
    elif "power_user" in permissions:
        llm_request.tools_dict.update(power_user_tools)

    # Everyone gets basic tools
    llm_request.tools_dict.update(basic_tools)

    return None

agent = LlmAgent(
    name="permission_agent",
    instruction="Help the user with available tools",
    before_model_callback=permission_based_tools,
)
```

#### Example 2: Context-Aware Tool Selection

```python
def context_aware_tools(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Select tools based on conversation context."""
    conversation_topic = callback_context.session.state.get("topic", "general")

    # Clear existing tools
    llm_request.tools_dict.clear()

    # Load topic-specific tools
    if conversation_topic == "weather":
        llm_request.tools_dict["get_weather"] = FunctionTool(func=get_weather)
        llm_request.tools_dict["get_forecast"] = FunctionTool(func=get_forecast)

    elif conversation_topic == "finance":
        llm_request.tools_dict["get_stock_price"] = FunctionTool(func=get_stock_price)
        llm_request.tools_dict["calculate_returns"] = FunctionTool(func=calculate_returns)

    elif conversation_topic == "travel":
        llm_request.tools_dict["search_flights"] = FunctionTool(func=search_flights)
        llm_request.tools_dict["book_hotel"] = FunctionTool(func=book_hotel)

    # Always include help tool
    llm_request.tools_dict["help"] = FunctionTool(func=show_help)

    return None

agent = LlmAgent(
    name="topic_agent",
    instruction="Help with {topic} related queries",
    before_model_callback=context_aware_tools,
)
```

#### Example 3: Dynamic Prompt Injection

```python
def dynamic_prompt_injection(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Inject additional context into the prompt."""
    # Get runtime data
    user_history = callback_context.session.state.get("user_history", [])
    current_page = callback_context.session.state.get("current_page", "home")

    # Build dynamic context
    context = f"""Current Context:
- User is on page: {current_page}
- Recent actions: {', '.join(user_history[-3:])}
- Session duration: {callback_context.session.state.get('duration', '0')} minutes

Use this context to provide more relevant responses."""

    # Inject as system instruction
    llm_request.append_instructions([context])

    return None

agent = LlmAgent(
    name="context_agent",
    instruction="You are a helpful assistant",
    before_model_callback=dynamic_prompt_injection,
)
```

#### Example 4: A/B Testing with Dynamic Config

```python
def ab_test_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """A/B test different model configurations."""
    user_id = callback_context.session.state.get("user_id", "")

    # Simple hash-based A/B split
    if hash(user_id) % 2 == 0:
        # Group A: Creative
        llm_request.config.temperature = 0.9
        llm_request.config.top_p = 0.95
        callback_context.session.state["ab_group"] = "creative"
    else:
        # Group B: Precise
        llm_request.config.temperature = 0.3
        llm_request.config.top_p = 0.5
        callback_context.session.state["ab_group"] = "precise"

    return None

agent = LlmAgent(
    name="ab_test_agent",
    instruction="Respond to user queries",
    before_model_callback=ab_test_callback,
)
```

#### Example 5: Rate Limiting Tool Access

```python
def rate_limited_tools(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Limit expensive tool usage based on rate limits."""
    tool_usage = callback_context.session.state.get("tool_usage", {})
    expensive_tool_count = tool_usage.get("expensive_api_call", 0)

    # Remove expensive tool if limit reached
    if expensive_tool_count >= 5:
        if "expensive_api_call" in llm_request.tools_dict:
            del llm_request.tools_dict["expensive_api_call"]

        # Add explanation to context
        llm_request.append_instructions([
            "Note: expensive_api_call tool is unavailable (rate limit reached)"
        ])

    return None

agent = LlmAgent(
    name="rate_limited_agent",
    instruction="Help the user",
    tools=[FunctionTool(func=expensive_api_call)],
    before_model_callback=rate_limited_tools,
)
```

---

## Combining Both Approaches

You can use both `InstructionProvider` and `before_model_callback` together:

```python
def dynamic_instruction(ctx: ReadonlyContext) -> str:
    """Generate base instruction."""
    mode = ctx.session.state.get("mode", "normal")
    return f"You are in {mode} mode."

def enhance_with_tools(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Add tools based on mode."""
    mode = callback_context.session.state.get("mode", "normal")

    if mode == "debug":
        llm_request.tools_dict["debug_info"] = FunctionTool(func=get_debug_info)

    return None

agent = LlmAgent(
    name="flexible_agent",
    instruction=dynamic_instruction,  # Dynamic prompt
    before_model_callback=enhance_with_tools,  # Dynamic tools
)
```

---

## Complete Example: Multi-Tenant Agent

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool


# Define tools
def basic_search(query: str) -> str:
    """Basic search available to all."""
    return f"Results for: {query}"

def advanced_analytics(data: str) -> str:
    """Premium analytics tool."""
    return f"Advanced analytics for: {data}"

def admin_delete(item_id: str) -> str:
    """Admin-only delete function."""
    return f"Deleted item: {item_id}"


# Dynamic instruction based on tenant
def tenant_instruction(ctx: ReadonlyContext) -> str:
    """Customize instruction per tenant."""
    tenant_id = ctx.session.state.get("tenant_id", "default")
    tier = ctx.session.state.get("tenant_tier", "free")

    base = f"You are assisting tenant '{tenant_id}'.\n"

    if tier == "enterprise":
        base += "Provide expert-level support with full feature access."
    elif tier == "premium":
        base += "Provide enhanced support with analytics access."
    else:
        base += "Provide standard support."

    return base


# Dynamic tools based on tenant tier
def tenant_tools(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Load tools based on tenant tier and permissions."""
    tier = callback_context.session.state.get("tenant_tier", "free")
    permissions = callback_context.session.state.get("permissions", [])

    # Start fresh
    llm_request.tools_dict.clear()

    # Basic tools for everyone
    llm_request.tools_dict["basic_search"] = FunctionTool(func=basic_search)

    # Premium tools
    if tier in ["premium", "enterprise"]:
        llm_request.tools_dict["advanced_analytics"] = FunctionTool(
            func=advanced_analytics
        )

    # Admin tools
    if "admin" in permissions:
        llm_request.tools_dict["admin_delete"] = FunctionTool(func=admin_delete)

    # Log available tools
    available = list(llm_request.tools_dict.keys())
    callback_context.session.state["available_tools"] = available

    return None


# Create multi-tenant agent
multi_tenant_agent = LlmAgent(
    name="multi_tenant_agent",
    instruction=tenant_instruction,
    before_model_callback=tenant_tools,
)


# Usage example
async def use_multi_tenant():
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    session_service = InMemorySessionService()

    # Tenant 1: Free tier user
    await session_service.create_session(
        app_name="MultiTenant",
        user_id="tenant1",
        session_id="s1",
        state={
            "tenant_id": "acme_corp",
            "tenant_tier": "free",
            "permissions": ["user"],
        }
    )

    # Tenant 2: Enterprise with admin
    await session_service.create_session(
        app_name="MultiTenant",
        user_id="tenant2",
        session_id="s2",
        state={
            "tenant_id": "bigco",
            "tenant_tier": "enterprise",
            "permissions": ["admin", "user"],
        }
    )

    runner = Runner(
        app_name="MultiTenant",
        agent=multi_tenant_agent,
        session_service=session_service
    )

    # Tenant 1 gets basic tools only
    events1 = runner.run_async(
        user_id="tenant1",
        session_id="s1",
        new_message=types.Content(role='user', parts=[types.Part(text="What can you do?")])
    )

    # Tenant 2 gets full access
    events2 = runner.run_async(
        user_id="tenant2",
        session_id="s2",
        new_message=types.Content(role='user', parts=[types.Part(text="What can you do?")])
    )
```

---

## Advanced: BaseToolset for Dynamic Tools

For more sophisticated tool loading, create a custom `BaseToolset`:

```python
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.readonly_context import ReadonlyContext

class DynamicToolset(BaseToolset):
    """Toolset that loads tools based on context."""

    async def get_tools(self, ctx: ReadonlyContext) -> list[BaseTool]:
        """Return tools based on session state."""
        user_role = ctx.session.state.get("role", "guest")

        tools = []

        # Everyone gets basic tools
        tools.append(FunctionTool(func=basic_function))

        # Role-specific tools
        if user_role == "admin":
            tools.append(FunctionTool(func=admin_function))
        elif user_role == "user":
            tools.append(FunctionTool(func=user_function))

        return tools

# Use in agent
agent = LlmAgent(
    name="dynamic_toolset_agent",
    instruction="Help the user",
    tools=[DynamicToolset()],  # Toolset, not individual tools
)
```

---

## Performance Considerations

**TL;DR: Callback overhead is negligible (~1-5ms) compared to LLM latency (~2000ms).**

### Overhead Analysis

```
Total Request Time: ~2000ms
├── before_model_callback: ~5ms     (0.25%)  ← Your callback here
├── Network + serialization: ~60ms  (3%)
└── LLM inference: ~1900ms          (95%)
```

**Key insight**: Your callback is **400x faster** than the LLM call!

### When Overhead Matters

✅ **Negligible (<5ms)**:
- State lookups: `ctx.session.state.get("key")`
- Conditional logic: `if tier == "premium"`
- Tool dictionary updates: `request.tools_dict.update(tools)`

⚠️ **Noticeable (10-50ms)**:
- Complex algorithms
- Large data processing
- Many nested loops

❌ **Unacceptable (>50ms)**:
- Database queries
- Network/API calls
- File I/O

### Solution: Pre-Fetch in DummyAgent

```python
# ❌ BAD: Slow callback
def slow_callback(ctx, request):
    tools = requests.get("https://api.example.com/tools").json()  # 500ms!
    request.tools_dict.update(tools)
    return None

# ✅ GOOD: Pre-fetch in DummyAgent
def fetch_tools(ctx):
    """Runs BEFORE the LLM agent."""
    tools = requests.get("https://api.example.com/tools").json()
    ctx.session.state["tools"] = tools
    return "Tools loaded"

fetcher = DummyAgent(name="fetcher", logic_function=fetch_tools)

def fast_callback(ctx, request):
    tools = ctx.session.state.get("tools")  # Instant!
    request.tools_dict.update(tools)
    return None

# Workflow: Fetch once, then use
workflow = SequentialAgent(sub_agents=[fetcher, llm_agent])
```

**See `performance_considerations.md` for detailed analysis and benchmarks.**

---

## Best Practices

### ✅ DO:

1. **Cache expensive computations**
   ```python
   _tool_cache = {}

   def cached_tools(ctx):
       user_id = ctx.session.state.get("user_id")
       if user_id in _tool_cache:
           return _tool_cache[user_id]

       tools = compute_expensive_tools(user_id)
       _tool_cache[user_id] = tools
       return tools
   ```

2. **Log dynamic decisions for debugging**
   ```python
   def logged_callback(callback_context, llm_request):
       tools_added = ["tool1", "tool2"]
       callback_context.session.state["tools_loaded"] = tools_added
       print(f"Loaded tools: {tools_added}")
       return None
   ```

3. **Handle missing state gracefully**
   ```python
   def safe_instruction(ctx):
       tier = ctx.session.state.get("tier", "default")
       # Always provide a valid instruction
       return INSTRUCTIONS.get(tier, DEFAULT_INSTRUCTION)
   ```

### ❌ DON'T:

1. **Don't make slow API calls in callbacks**
   ```python
   # BAD - blocks the LLM call
   def slow_callback(ctx, request):
       data = requests.get("https://slow-api.com").json()  # SLOW!
       return None

   # GOOD - pre-fetch in a dummy agent
   ```

2. **Don't modify state in before_model_callback**
   ```python
   # BAD - callbacks should be read-only
   def bad_callback(callback_context, llm_request):
       callback_context.session.state["modified"] = True  # Don't do this
       return None
   ```

3. **Don't forget to return None**
   ```python
   # BAD - returning a value short-circuits the LLM call
   def bad_callback(ctx, request):
       request.tools_dict.clear()
       return "something"  # WRONG! Should return None

   # GOOD
   def good_callback(ctx, request):
       request.tools_dict.clear()
       return None  # Continue to LLM
   ```

---

## Multi-Turn Durability

### Yes, Dynamic Tools and Prompts ARE Durable Across Turns!

Both `InstructionProvider` and `before_model_callback` are **re-evaluated on EVERY turn** based on the current session state.

```python
# Turn 1: User is free tier
session.state["user_tier"] = "free"
# → InstructionProvider returns "You are a FREE assistant"
# → before_model_callback adds basic_tool only

# Turn 2: Upgrade user mid-conversation
session.state["user_tier"] = "premium"
# → InstructionProvider returns "You are a PREMIUM assistant"
# → before_model_callback adds basic_tool + premium_tool

# Turn 3: Can downgrade too
session.state["user_tier"] = "free"
# → Back to free tier behavior
```

### How It Works

1. **Session state persists** between turns
2. **Dynamic functions re-read** state on each turn
3. **Changes take effect immediately** on the next turn

### Important Notes

**✅ State Changes Are Immediate**
```python
# During conversation:
session = await session_service.get_session(app_name, user_id, session_id)
session.state["feature_enabled"] = True

# Next turn:
# - InstructionProvider sees feature_enabled=True
# - before_model_callback sees feature_enabled=True
# - Agent behavior changes immediately
```

**⚠️ State Must Be Persisted**
- If using `InMemorySessionService`, state is lost when server restarts
- Use persistent session service for production (Firestore, etc.)
- State changes must be saved to session service

**📊 State Is Shared Across All Callbacks**
- `InstructionProvider` gets `ReadonlyContext` with `session.state`
- `before_model_callback` gets `CallbackContext` with `session.state`
- Both read from the same session state object

### Use Cases for Multi-Turn Dynamic Behavior

1. **Progressive Feature Unlocking**
   ```python
   # Unlock features as conversation progresses
   turn_count = ctx.session.state.get("turn_count", 0)
   if turn_count > 5:
       # User has been engaged, unlock advanced features
       llm_request.tools_dict["advanced_tool"] = ...
   ```

2. **Context-Aware Tool Loading**
   ```python
   # Load tools based on what was discussed
   if "weather" in ctx.session.state.get("topics", []):
       llm_request.tools_dict["get_weather"] = ...
   ```

3. **Mid-Conversation Upgrades**
   ```python
   # User purchases premium mid-conversation
   # Next turn automatically gets premium features
   session.state["tier"] = "premium"
   ```

4. **Rate Limiting**
   ```python
   # Disable expensive tools after limit reached
   api_calls = ctx.session.state.get("api_calls", 0)
   if api_calls >= 10:
       if "expensive_api" in llm_request.tools_dict:
           del llm_request.tools_dict["expensive_api"]
   ```

5. **Adaptive Difficulty**
   ```python
   # Simplify if user seems confused
   confusion_score = ctx.session.state.get("confusion_score", 0)
   if confusion_score > 5:
       return "Explain things very simply, step by step."
   ```

### Testing Multi-Turn Behavior

See `test_dynamic_multiturn.py` for a complete example that demonstrates:
- State changes between turns
- Dynamic tool addition/removal
- Dynamic instruction changes
- Verification that changes take effect immediately

---

## Summary

ADK supports dynamic behavior through:

1. **InstructionProvider** - Functions that generate prompts based on `ReadonlyContext`
   - Access to session state
   - Can be sync or async
   - Called every time agent runs

2. **before_model_callback** - Intercepts LLM requests
   - Modify `tools_dict` for dynamic tools
   - Modify `contents` for context injection
   - Modify `config` for A/B testing
   - Access to `CallbackContext` with session state

Use these together to create:
- Multi-tenant agents
- Permission-based tool access
- Context-aware behavior
- A/B testing
- Rate limiting
- Dynamic routing

**Remember**: Dynamic doesn't mean random - it means adapting to context while remaining deterministic given the same input state!

---

## Sources

- [Agent Development Kit Documentation](https://google.github.io/adk-docs/)
- [Custom Tools for ADK](https://google.github.io/adk-docs/tools-custom/)
- [Runtime Config Documentation](https://google.github.io/adk-docs/runtime/runconfig/)
