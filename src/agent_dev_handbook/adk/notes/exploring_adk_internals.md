# Exploring ADK Internals

A practical guide to investigating any part of the Google ADK library.

---

## Quick Reference

| What You Want | Where to Look |
|---------------|---------------|
| Type signatures | `.venv/.../google/adk/agents/*.py` - look for `TypeAlias` |
| Class definitions | `.venv/.../google/adk/` - find the module, read the class |
| Execution flow | `.venv/.../google/adk/flows/` - trace how things are called |
| Available methods | `dir(obj)` or `inspect.getmembers(obj)` |
| Method signatures | `inspect.signature(method)` |
| Source location | `inspect.getfile(cls)` |

---

## Step 1: Locate the Installed Package

```bash
# Find where ADK is installed
python -c "import google.adk; print(google.adk.__file__)"

# Typical location
.venv/lib/python3.12/site-packages/google/adk/
```

**Key directories:**

```
google/adk/
├── agents/           # Agent classes (LlmAgent, BaseAgent, etc.)
├── models/           # LlmRequest, LlmResponse, etc.
├── flows/            # Execution logic (where things actually happen)
├── tools/            # Tool definitions
├── sessions/         # Session management
├── events/           # Event system
└── runners/          # Runner implementations
```

---

## Step 2: Find Type Definitions

Search for `TypeAlias` to find expected types:

```bash
# Search for type definitions
grep -r "TypeAlias" .venv/lib/python3.12/site-packages/google/adk/
```

Example from `agents/llm_agent.py`:

```python
BeforeModelCallback: TypeAlias = Callable[
    [CallbackContext, LlmRequest],
    Union[Awaitable[Optional[LlmResponse]], Optional[LlmResponse]],
]
```

This tells you exactly what parameters a callback receives and what it should return.

---

## Step 3: Inspect Classes at Runtime

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
import inspect

# List all public attributes/methods
public_members = [m for m in dir(LlmRequest) if not m.startswith('_')]
print(public_members)

# Get method signature
print(inspect.signature(LlmRequest.append_instructions))

# Check if method is async
print(inspect.iscoroutinefunction(CallbackContext.load_artifact))

# Find source file
print(inspect.getfile(LlmRequest))

# Read actual source code
print(inspect.getsource(LlmRequest.append_instructions))
```

---

## Step 4: Trace Execution Flow

To understand *when* something is called, search the `flows/` directory:

```bash
# Find where callbacks are invoked
grep -r "before_model_callback" .venv/.../google/adk/flows/
```

Look for patterns like:
- `await callback(...)` - where it's called
- `if result is not None` - how return values are handled
- `for callback in callbacks` - iteration order

---

## Step 5: Build a Quick Inspector

Create a simple utility to dump object info:

```python
import inspect

def inspect_object(obj, name="Object"):
    """Quick inspection of any ADK object."""
    print(f"\n{'='*50}")
    print(f"Class: {obj.__class__.__name__}")
    print(f"Module: {obj.__class__.__module__}")

    # Properties (non-callable attributes)
    print(f"\nProperties:")
    for attr in dir(obj):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(obj, attr)
            if not callable(val):
                print(f"  {attr}: {type(val).__name__}")
        except:
            pass

    # Methods
    print(f"\nMethods:")
    for attr in dir(obj):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(obj, attr)
            if callable(val):
                sig = str(inspect.signature(val))
                async_mark = " (async)" if inspect.iscoroutinefunction(val) else ""
                print(f"  {attr}{sig}{async_mark}")
        except:
            pass
```

---

## Common Investigation Patterns

### Pattern A: "What can I do with X?"

```python
# 1. Import the class
from google.adk.models.llm_request import LlmRequest

# 2. List methods
[m for m in dir(LlmRequest) if not m.startswith('_') and callable(getattr(LlmRequest, m, None))]

# 3. Check signatures
import inspect
inspect.signature(LlmRequest.append_tools)
```

### Pattern B: "What parameters does X receive?"

```bash
# Search for TypeAlias or function signatures
grep -r "def your_function_name" .venv/.../google/adk/
grep -r "YourType: TypeAlias" .venv/.../google/adk/
```

### Pattern C: "When is X called?"

```bash
# Search for invocations in flows/
grep -r "your_callback\|your_method" .venv/.../google/adk/flows/
```

### Pattern D: "What does X inherit from?"

```python
from google.adk.agents.callback_context import CallbackContext
print(CallbackContext.__mro__)  # Method Resolution Order
# Shows: CallbackContext -> ReadonlyContext -> object
```

---

## Useful Grep Patterns

```bash
# Find class definitions
grep -r "^class YourClass" .venv/.../google/adk/

# Find where exceptions are raised
grep -r "raise.*Error" .venv/.../google/adk/

# Find async methods
grep -r "async def" .venv/.../google/adk/agents/

# Find all TypeAlias definitions
grep -r ": TypeAlias" .venv/.../google/adk/

# Find dataclass definitions
grep -r "@dataclass" .venv/.../google/adk/
```

---

## Key Files by Topic

| Topic | Key Files |
|-------|-----------|
| Agent definitions | `agents/llm_agent.py`, `agents/base_agent.py` |
| Callbacks | `agents/callback_context.py`, `agents/llm_agent.py` |
| Request/Response | `models/llm_request.py`, `models/llm_response.py` |
| Execution flow | `flows/llm_flows/base_llm_flow.py` |
| Events | `events/event.py`, `events/event_actions.py` |
| Tools | `tools/base_tool.py`, `tools/function_tool.py` |
| Sessions | `sessions/session.py`, `sessions/state.py` |
| Invocation context | `agents/invocation_context.py` |

---

## Example: Investigating a New Feature

Say you want to understand how `output_key` works in `LlmAgent`:

```bash
# 1. Find where it's defined
grep -r "output_key" .venv/.../google/adk/agents/llm_agent.py

# 2. Find where it's used
grep -r "output_key" .venv/.../google/adk/flows/

# 3. Read the relevant code section
# Look for: ctx.session.state[self.output_key] = ...
```

You'll discover it saves the LLM response text to `session.state[output_key]`.

---

## Quick Python REPL Session

```python
# Start exploring any ADK component
import inspect
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext

# What does InvocationContext have?
for name, method in inspect.getmembers(InvocationContext, predicate=inspect.isfunction):
    if not name.startswith('_'):
        print(f"{name}: {inspect.signature(method)}")

# What's the inheritance chain?
print(LlmAgent.__mro__)

# Where's the source?
print(inspect.getfile(LlmAgent))
```

---

## Summary

1. **Find the source**: `.venv/.../google/adk/`
2. **Search for types**: `grep -r "TypeAlias"` or read class definitions
3. **Inspect at runtime**: `dir()`, `inspect.signature()`, `inspect.getsource()`
4. **Trace execution**: Look in `flows/` for where things are called
5. **Check inheritance**: `ClassName.__mro__` shows parent classes

This approach works for any Python library, not just ADK.
