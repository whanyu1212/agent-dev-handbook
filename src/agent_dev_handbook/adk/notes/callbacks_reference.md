# ADK Callbacks Reference

## Overview

ADK provides **callbacks** as hooks into the agent execution lifecycle. Callbacks allow you to:
- Observe agent behavior (logging, debugging, tracing)
- Modify requests before they reach the LLM
- Transform responses after they come back
- Short-circuit LLM calls entirely (caching, guardrails)
- Implement cross-cutting concerns (auth, rate limiting)

---

## Types of Callbacks

### 1. `before_model_callback`
Called **before** the LLM request is sent. Can modify or replace the request.

### 2. `after_model_callback`
Called **after** the LLM response is received. Can modify or replace the response.

---

## Type Signatures

```python
from typing import Callable, Union, Optional, Awaitable
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

# Single before_model_callback
_SingleBeforeModelCallback = Callable[
    [CallbackContext, LlmRequest],
    Union[Awaitable[Optional[LlmResponse]], Optional[LlmResponse]],
]

# Can be single callback or list of callbacks
BeforeModelCallback = Union[
    _SingleBeforeModelCallback,
    list[_SingleBeforeModelCallback],
]

# Single after_model_callback
_SingleAfterModelCallback = Callable[
    [CallbackContext, LlmResponse],
    Union[Awaitable[Optional[LlmResponse]], Optional[LlmResponse]],
]

# Can be single callback or list of callbacks
AfterModelCallback = Union[
    _SingleAfterModelCallback,
    list[_SingleAfterModelCallback],
]
```

**Key Points:**
- Callbacks can be **sync or async**
- Can return `None` to continue normal flow
- Can return `LlmResponse` to short-circuit/replace
- Can pass a **list** of callbacks (executed in order)

---

## CallbackContext

The `CallbackContext` object provides access to the current execution context.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `agent_name` | `str` | Name of the current agent |
| `invocation_id` | `str` | Unique ID for this invocation |
| `user_id` | `str` | User ID from the session |
| `session` | `Session` | Current session object |
| `state` | `State` | Delta-aware mutable state (tracks changes) |
| `user_content` | `Optional[types.Content]` | User content that triggered this invocation |
| `run_config` | `Optional[RunConfig]` | Runtime configuration |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load_artifact` | `async (filename: str, version: Optional[int]) -> Optional[types.Part]` | Load an artifact from storage |
| `save_artifact` | `async (filename: str, artifact: types.Part, custom_metadata: Optional[dict]) -> int` | Save an artifact to storage |
| `get_artifact_version` | `async (filename: str, version: Optional[int]) -> Optional[ArtifactVersion]` | Get artifact version metadata |
| `list_artifacts` | `async () -> list[str]` | List all artifacts |
| `save_credential` | `async (auth_config: AuthConfig) -> None` | Save authentication credential |
| `load_credential` | `async (auth_config: AuthConfig) -> Optional[AuthCredential]` | Load authentication credential |
| `add_session_to_memory` | `async () -> None` | Add current session to memory service |

### Accessing Session State

```python
def my_callback(ctx: CallbackContext, request: LlmRequest):
    # Read from session state
    user_tier = ctx.session.state.get("user_tier", "free")

    # Access state delta (tracks changes)
    ctx.state["new_key"] = "new_value"

    return None
```

### Dunder Methods

Available magic methods on `CallbackContext`:
- `__init__`, `__repr__`, `__str__`
- `__eq__`, `__ne__`, `__hash__`
- `__getattribute__`, `__setattr__`, `__delattr__`

---

## LlmRequest

The `LlmRequest` object represents the request being sent to the LLM.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `Optional[str]` | Model name/identifier |
| `contents` | `list[types.Content]` | Conversation contents to send |
| `config` | `types.GenerateContentConfig` | Generation configuration |
| `tools_dict` | `dict[str, BaseTool]` | Dictionary of available tools |
| `live_connect_config` | `types.LiveConnectConfig` | Live API configuration |
| `cache_config` | `Optional[ContextCacheConfig]` | Context caching configuration |
| `cache_metadata` | `Optional[CacheMetadata]` | Cache metadata from previous requests |
| `cacheable_contents_token_count` | `Optional[int]` | Token count from previous request |
| `previous_interaction_id` | `Optional[str]` | ID for stateful conversations |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `append_instructions` | `(instructions: Union[list[str], types.Content]) -> list[types.Content]` | Append instructions to the request |
| `append_tools` | `(tools: list[BaseTool]) -> None` | Add tools to the request |
| `set_output_schema` | `(base_model: type[BaseModel]) -> None` | Set structured output schema |

### Modifying Tools

```python
from google.adk.tools.function_tool import FunctionTool

def my_callback(ctx: CallbackContext, request: LlmRequest):
    # Clear all tools
    request.tools_dict.clear()

    # Add a tool
    request.tools_dict["my_tool"] = FunctionTool(func=my_function)

    # Remove a specific tool
    if "unwanted_tool" in request.tools_dict:
        del request.tools_dict["unwanted_tool"]

    # Get tool names
    tool_names = list(request.tools_dict.keys())

    return None
```

### Modifying Contents

```python
from google.genai import types

def my_callback(ctx: CallbackContext, request: LlmRequest):
    # Append additional context
    request.contents.append(
        types.Content(
            role='user',
            parts=[types.Part(text="Additional context here")]
        )
    )

    # Check existing contents
    for content in request.contents:
        print(f"Role: {content.role}, Parts: {len(content.parts)}")

    return None
```

### Modifying Config

```python
def my_callback(ctx: CallbackContext, request: LlmRequest):
    # Adjust generation parameters
    request.config.temperature = 0.7
    request.config.max_output_tokens = 1000
    request.config.top_p = 0.9
    request.config.top_k = 40

    return None
```

### Dunder Methods

Available magic methods on `LlmRequest`:
- `__init__`, `__repr__`, `__str__`
- `__eq__`, `__ne__`, `__hash__`
- `__getattribute__`, `__setattr__`, `__delattr__`

---

## LlmResponse

The `LlmResponse` object represents the response from the LLM.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model_version` | `Optional[str]` | Model version used |
| `content` | `Optional[types.Content]` | Response content |
| `partial` | `Optional[bool]` | Whether response is partial (streaming) |
| `turn_complete` | `Optional[bool]` | Whether turn is complete |
| `finish_reason` | `Optional[types.FinishReason]` | Why generation finished |
| `error_code` | `Optional[str]` | Error code if response is error |
| `error_message` | `Optional[str]` | Error message if response is error |
| `interrupted` | `Optional[bool]` | Whether LLM was interrupted |
| `grounding_metadata` | `Optional[types.GroundingMetadata]` | Grounding metadata |
| `usage_metadata` | `Optional[types.GenerateContentResponseUsageMetadata]` | Token usage info |
| `citation_metadata` | `Optional[types.CitationMetadata]` | Citation metadata |
| `input_transcription` | `Optional[types.Transcription]` | Audio input transcription |
| `output_transcription` | `Optional[types.Transcription]` | Audio output transcription |
| `avg_logprobs` | `Optional[float]` | Average log probability |
| `logprobs_result` | `Optional[types.LogprobsResult]` | Detailed log probabilities |
| `cache_metadata` | `Optional[CacheMetadata]` | Cache info if caching was used |
| `interaction_id` | `Optional[str]` | Interactions API ID |
| `custom_metadata` | `Optional[dict[str, Any]]` | Custom metadata |

### Accessing Content

```python
def my_after_callback(ctx: CallbackContext, response: LlmResponse):
    if response.content and response.content.parts:
        for part in response.content.parts:
            # Text response
            if hasattr(part, 'text') and part.text:
                print(f"Text: {part.text}")

            # Function call
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                print(f"Function: {fc.name}, Args: {fc.args}")

            # Function response
            if hasattr(part, 'function_response') and part.function_response:
                fr = part.function_response
                print(f"Response for: {fr.name}")

    return None
```

### Accessing Usage Metadata

```python
def my_after_callback(ctx: CallbackContext, response: LlmResponse):
    if response.usage_metadata:
        usage = response.usage_metadata
        print(f"Prompt tokens: {usage.prompt_token_count}")
        print(f"Response tokens: {usage.candidates_token_count}")
        print(f"Total tokens: {usage.total_token_count}")
        print(f"Cached tokens: {usage.cached_content_token_count}")

    return None
```

### Checking for Errors

```python
def my_after_callback(ctx: CallbackContext, response: LlmResponse):
    if response.error_code:
        print(f"Error: {response.error_code} - {response.error_message}")
        # Could return a fallback response here
        return LlmResponse(
            content=types.Content(
                role='model',
                parts=[types.Part(text="Sorry, an error occurred.")]
            )
        )

    return None
```

### Dunder Methods

Available magic methods on `LlmResponse`:
- `__init__`, `__repr__`, `__str__`
- `__eq__`, `__ne__`, `__hash__`
- `__getattribute__`, `__setattr__`, `__delattr__`

---

## Execution Flow

The callback execution flow in ADK:

```
User Message
     │
     ▼
┌─────────────────────┐
│  Plugin Callbacks   │  (from registered plugins)
│  before_model_cb    │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Agent Callbacks    │  (from agent definition)
│  before_model_cb    │
└─────────────────────┘
     │
     ├──── Returns LlmResponse? ──► Skip LLM, use response
     │
     ▼
┌─────────────────────┐
│     LLM Call        │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Plugin Callbacks   │  (from registered plugins)
│  after_model_cb     │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Agent Callbacks    │  (from agent definition)
│  after_model_cb     │
└─────────────────────┘
     │
     ▼
Response to User
```

**Key Points:**
1. Plugin callbacks run first, then agent callbacks
2. `before_model_callback` returning `LlmResponse` skips the LLM call entirely
3. `after_model_callback` returning `LlmResponse` replaces the original response
4. Multiple callbacks in a list execute in order

---

## Common Patterns

### Pattern 1: Logging/Debugging

```python
def logging_before_callback(ctx: CallbackContext, request: LlmRequest):
    """Log request details for debugging."""
    print(f"[{ctx.agent_name}] Request to model: {request.model}")
    print(f"  Tools: {list(request.tools_dict.keys())}")
    print(f"  Contents: {len(request.contents)} messages")
    print(f"  State: {dict(ctx.session.state)}")
    return None

def logging_after_callback(ctx: CallbackContext, response: LlmResponse):
    """Log response details for debugging."""
    print(f"[{ctx.agent_name}] Response received")
    print(f"  Model version: {response.model_version}")
    print(f"  Finish reason: {response.finish_reason}")
    if response.usage_metadata:
        print(f"  Tokens: {response.usage_metadata.total_token_count}")
    return None
```

### Pattern 2: Caching (Short-circuit LLM)

```python
_cache = {}

def cache_before_callback(ctx: CallbackContext, request: LlmRequest):
    """Return cached response if available."""
    # Create cache key from last user message
    if request.contents:
        last_content = request.contents[-1]
        if last_content.parts and last_content.parts[0].text:
            cache_key = last_content.parts[0].text

            if cache_key in _cache:
                print("Cache hit!")
                return _cache[cache_key]  # Skip LLM

    return None  # Cache miss, continue to LLM

def cache_after_callback(ctx: CallbackContext, response: LlmResponse):
    """Cache the response."""
    # Store in cache for future requests
    # (You'd need to pass the cache key somehow)
    return None
```

### Pattern 3: Guardrails / Content Filtering

```python
from google.genai import types

BLOCKED_TERMS = ["forbidden", "inappropriate"]

def guardrail_before_callback(ctx: CallbackContext, request: LlmRequest):
    """Block requests containing forbidden content."""
    for content in request.contents:
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                for term in BLOCKED_TERMS:
                    if term.lower() in part.text.lower():
                        # Return a blocking response
                        return LlmResponse(
                            content=types.Content(
                                role='model',
                                parts=[types.Part(text="I cannot process this request.")]
                            )
                        )
    return None

def guardrail_after_callback(ctx: CallbackContext, response: LlmResponse):
    """Filter inappropriate responses."""
    if response.content and response.content.parts:
        for part in response.content.parts:
            if hasattr(part, 'text') and part.text:
                for term in BLOCKED_TERMS:
                    if term.lower() in part.text.lower():
                        # Replace with safe response
                        return LlmResponse(
                            content=types.Content(
                                role='model',
                                parts=[types.Part(text="[Content filtered]")]
                            )
                        )
    return None
```

### Pattern 4: Dynamic Tool Loading

```python
from google.adk.tools.function_tool import FunctionTool

def basic_tool():
    """Basic tool for all users."""
    return "Basic result"

def premium_tool():
    """Premium tool for paid users."""
    return "Premium result"

def dynamic_tools_callback(ctx: CallbackContext, request: LlmRequest):
    """Load tools based on user tier."""
    tier = ctx.session.state.get("user_tier", "free")

    # Clear and rebuild tools
    request.tools_dict.clear()
    request.tools_dict["basic_tool"] = FunctionTool(func=basic_tool)

    if tier == "premium":
        request.tools_dict["premium_tool"] = FunctionTool(func=premium_tool)

    return None
```

### Pattern 5: Rate Limiting

```python
def rate_limit_callback(ctx: CallbackContext, request: LlmRequest):
    """Enforce rate limits."""
    call_count = ctx.session.state.get("llm_calls", 0)
    max_calls = ctx.session.state.get("max_calls", 10)

    if call_count >= max_calls:
        return LlmResponse(
            content=types.Content(
                role='model',
                parts=[types.Part(text="Rate limit exceeded. Please try again later.")]
            ),
            error_code="RATE_LIMITED",
            error_message="Maximum calls exceeded"
        )

    # Increment counter
    ctx.state["llm_calls"] = call_count + 1
    return None
```

### Pattern 6: Token Usage Tracking

```python
def track_usage_callback(ctx: CallbackContext, response: LlmResponse):
    """Track token usage across session."""
    if response.usage_metadata:
        # Get current totals
        total_prompt = ctx.session.state.get("total_prompt_tokens", 0)
        total_response = ctx.session.state.get("total_response_tokens", 0)

        # Update totals
        ctx.state["total_prompt_tokens"] = total_prompt + (
            response.usage_metadata.prompt_token_count or 0
        )
        ctx.state["total_response_tokens"] = total_response + (
            response.usage_metadata.candidates_token_count or 0
        )

        print(f"Session token usage: {ctx.state['total_prompt_tokens']} prompt, "
              f"{ctx.state['total_response_tokens']} response")

    return None
```

### Pattern 7: Response Transformation

```python
def transform_response_callback(ctx: CallbackContext, response: LlmResponse):
    """Transform or augment responses."""
    if response.content and response.content.parts:
        new_parts = []
        for part in response.content.parts:
            if hasattr(part, 'text') and part.text:
                # Add disclaimer to all text responses
                transformed_text = part.text + "\n\n---\n*This is an AI-generated response.*"
                new_parts.append(types.Part(text=transformed_text))
            else:
                new_parts.append(part)

        return LlmResponse(
            content=types.Content(
                role=response.content.role,
                parts=new_parts
            ),
            model_version=response.model_version,
            finish_reason=response.finish_reason,
            usage_metadata=response.usage_metadata,
        )

    return None
```

---

## Multiple Callbacks

You can register multiple callbacks as a list:

```python
agent = LlmAgent(
    name="multi_callback_agent",
    instruction="You are a helpful assistant.",
    before_model_callback=[
        logging_before_callback,
        guardrail_before_callback,
        dynamic_tools_callback,
    ],
    after_model_callback=[
        logging_after_callback,
        track_usage_callback,
        transform_response_callback,
    ],
)
```

Callbacks execute in order. If any returns an `LlmResponse`:
- For `before_model_callback`: Remaining callbacks and LLM call are skipped
- For `after_model_callback`: Remaining callbacks still run with the new response

---

## Async Callbacks

Callbacks can be async for I/O operations:

```python
async def async_before_callback(ctx: CallbackContext, request: LlmRequest):
    """Async callback for database lookup."""
    import aiohttp

    user_id = ctx.user_id
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/user/{user_id}/tools") as resp:
            tools_config = await resp.json()

    # Configure tools based on response
    for tool_name in tools_config.get("enabled_tools", []):
        # Add tools dynamically
        pass

    return None

async def async_after_callback(ctx: CallbackContext, response: LlmResponse):
    """Async callback for logging to external service."""
    import aiohttp

    log_data = {
        "agent": ctx.agent_name,
        "model": response.model_version,
        "tokens": response.usage_metadata.total_token_count if response.usage_metadata else None,
    }

    async with aiohttp.ClientSession() as session:
        await session.post("https://logging.example.com/log", json=log_data)

    return None
```

**Performance Note:** Avoid slow async operations in callbacks when possible. Consider pre-fetching data in a `DummyAgent` before the LLM agent runs.

---

## Inspection Utility

For debugging and understanding callbacks, use the `callback_inspector` utility:

```python
from agent_dev_handbook.adk.examples.callback_inspector import (
    create_before_model_inspector,
    create_after_model_inspector,
    inspect_callback_context,
    inspect_llm_request,
    inspect_llm_response,
)

# Create verbose inspectors
agent = LlmAgent(
    name="debug_agent",
    instruction="You are helpful.",
    before_model_callback=create_before_model_inspector(
        verbose=True,
        log_state=True,
        log_tools=True,
        log_contents=True,
    ),
    after_model_callback=create_after_model_inspector(
        verbose=True,
        log_content=True,
        log_usage=True,
    ),
)

# Or inspect manually in your own callback
def my_debug_callback(ctx: CallbackContext, request: LlmRequest):
    ctx_info = inspect_callback_context(ctx)
    req_info = inspect_llm_request(request)

    print(f"Context methods: {[m['name'] for m in ctx_info['methods']]}")
    print(f"Request tools: {list(req_info['tools'].keys())}")

    return None
```

---

## Best Practices

### Do:

1. **Return `None` to continue normal flow**
   ```python
   def good_callback(ctx, request):
       # Modify request
       request.tools_dict["new_tool"] = ...
       return None  # Continue to LLM
   ```

2. **Use state for cross-callback communication**
   ```python
   def before_cb(ctx, request):
       ctx.state["request_timestamp"] = time.time()
       return None

   def after_cb(ctx, response):
       start = ctx.session.state.get("request_timestamp")
       latency = time.time() - start
       print(f"Latency: {latency}s")
       return None
   ```

3. **Handle missing state gracefully**
   ```python
   def safe_callback(ctx, request):
       tier = ctx.session.state.get("tier", "default")  # Always provide default
       return None
   ```

4. **Log decisions for debugging**
   ```python
   def logged_callback(ctx, request):
       tools_added = ["tool1", "tool2"]
       print(f"[{ctx.agent_name}] Added tools: {tools_added}")
       return None
   ```

### Don't:

1. **Don't make slow I/O calls in callbacks**
   ```python
   # BAD - blocks LLM call
   def slow_callback(ctx, request):
       data = requests.get("https://slow-api.com").json()  # 500ms!
       return None

   # GOOD - pre-fetch in DummyAgent
   ```

2. **Don't accidentally return a value when you want to continue**
   ```python
   # BAD - this skips the LLM call!
   def bad_callback(ctx, request):
       return "some value"

   # GOOD
   def good_callback(ctx, request):
       return None
   ```

3. **Don't forget that callbacks run on every turn**
   ```python
   # Be aware this runs every turn, not just once
   def callback(ctx, request):
       print("This prints every single turn!")
       return None
   ```

---

## Sources

- [ADK Callbacks Documentation](https://google.github.io/adk-docs/callbacks/)
- [Types of Callbacks](https://google.github.io/adk-docs/callbacks/types-of-callbacks/)
- [Design Patterns and Best Practices](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/)
- Source: `google/adk/agents/llm_agent.py`
- Source: `google/adk/agents/callback_context.py`
- Source: `google/adk/models/llm_request.py`
- Source: `google/adk/models/llm_response.py`
- Source: `google/adk/flows/llm_flows/base_llm_flow.py`
