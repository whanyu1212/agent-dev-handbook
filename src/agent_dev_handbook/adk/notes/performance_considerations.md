# Performance Considerations for Dynamic Tools and Prompts

## TL;DR

**Yes, `before_model_callback` introduces overhead, but it's typically negligible compared to LLM latency.**

- ✅ **Callback execution**: ~1-10ms for simple logic
- ⚠️ **LLM API call**: 500-5000ms (100-1000x slower)
- ❌ **Avoid**: Network calls, heavy computation, blocking I/O in callbacks

**Bottom line**: Callback overhead is insignificant unless you do something expensive inside it.

---

## Performance Breakdown

### Where Time Is Spent in an LLM Agent Call

```
Total Request Time: ~2000ms
├── before_model_callback: ~5ms          (0.25% of total)
├── Request serialization: ~10ms         (0.5% of total)
├── Network round-trip: ~50ms            (2.5% of total)
├── LLM inference: ~1900ms               (95% of total)
├── Response parsing: ~10ms              (0.5% of total)
└── after_model_callback: ~5ms           (0.25% of total)
```

**Key insight**: The callback overhead (5ms) is **380x faster** than LLM inference (1900ms).

---

## Overhead Sources

### 1. Callback Function Execution

**Negligible for typical use cases:**

```python
# Fast (~0.1ms) - Simple state lookup
def fast_callback(ctx, request):
    tier = ctx.session.state.get("tier")
    if tier == "premium":
        request.tools_dict["premium_tool"] = tool
    return None
```

**Still acceptable (~1-5ms) - Tool dictionary manipulation:**

```python
# Medium (~2ms) - Loading multiple tools
def medium_callback(ctx, request):
    permissions = ctx.session.state.get("permissions", [])

    # Clear and rebuild (~1ms for 10 tools)
    request.tools_dict.clear()

    for perm in permissions:
        if perm == "admin":
            request.tools_dict.update(admin_tools)  # ~10 tools
        elif perm == "user":
            request.tools_dict.update(user_tools)   # ~5 tools

    return None
```

**Problematic (>100ms) - Expensive operations:**

```python
# SLOW (>500ms) - Network I/O
def slow_callback(ctx, request):
    # ❌ BAD: Blocks the entire request
    user_data = requests.get("https://api.example.com/user").json()  # 500ms!

    # ❌ BAD: Database query in callback
    tools = db.query("SELECT * FROM user_tools WHERE user_id = ?")  # 100ms!

    return None
```

---

## Benchmark: Callback Overhead

### Simple Callback (Best Case)

```python
import time

def simple_callback(ctx, request):
    tier = ctx.session.state.get("tier")
    if tier == "premium":
        request.tools_dict["tool1"] = tool1
    return None

# Benchmark
start = time.perf_counter()
for _ in range(1000):
    simple_callback(mock_ctx, mock_request)
end = time.perf_counter()

avg_time = (end - start) / 1000
print(f"Average: {avg_time * 1000:.3f}ms")
# Result: ~0.05ms per call
```

**Cost**: 0.05ms (essentially free)

### Complex Callback (Realistic Case)

```python
def complex_callback(ctx, request):
    # Multiple state lookups
    tier = ctx.session.state.get("tier")
    permissions = ctx.session.state.get("permissions", [])
    features = ctx.session.state.get("enabled_features", [])

    # Clear existing tools
    request.tools_dict.clear()

    # Conditional tool loading (20 tools)
    if tier == "premium":
        for tool_name, tool in premium_tools.items():
            request.tools_dict[tool_name] = tool

    if "admin" in permissions:
        for tool_name, tool in admin_tools.items():
            request.tools_dict[tool_name] = tool

    # Feature flags
    for feature in features:
        if feature in feature_tools:
            request.tools_dict[feature] = feature_tools[feature]

    return None

# Benchmark: ~2ms per call
```

**Cost**: 2ms (still negligible vs 2000ms LLM call)

### Expensive Callback (Anti-Pattern)

```python
def expensive_callback(ctx, request):
    # ❌ Database query
    tools = db.query("SELECT * FROM tools")  # 50ms

    # ❌ Network call
    config = requests.get("https://api.example.com/config").json()  # 500ms

    # ❌ Heavy computation
    for i in range(1000000):
        complex_calculation()  # 100ms

    return None

# Total: ~650ms (now significant!)
```

**Cost**: 650ms (26% of total request time - **BAD!**)

---

## Performance Impact by Use Case

### Use Case 1: Permission-Based Tools (FAST ✅)

```python
def permission_tools(ctx, request):
    permissions = ctx.session.state.get("permissions", [])

    if "admin" in permissions:
        request.tools_dict.update(admin_tools)

    return None

# Overhead: ~0.5ms
# Impact: 0.025% of total request time
# Verdict: ✅ No measurable impact
```

### Use Case 2: Dynamic Instruction (FAST ✅)

```python
def dynamic_instruction(ctx):
    role = ctx.session.state.get("role")
    return INSTRUCTION_TEMPLATES[role]  # Dict lookup

# Overhead: ~0.1ms
# Impact: 0.005% of total request time
# Verdict: ✅ No measurable impact
```

### Use Case 3: Tool Registry Lookup (ACCEPTABLE ⚠️)

```python
def registry_lookup(ctx, request):
    user_id = ctx.session.state.get("user_id")

    # In-memory cache lookup
    tools = tool_registry.get(user_id)  # ~1ms for complex lookup

    request.tools_dict.update(tools)
    return None

# Overhead: ~2ms
# Impact: 0.1% of total request time
# Verdict: ⚠️ Acceptable, but watch for cache misses
```

### Use Case 4: Database Query (SLOW ❌)

```python
def db_query(ctx, request):
    user_id = ctx.session.state.get("user_id")

    # ❌ BAD: Blocks for DB query
    tools = db.query(f"SELECT * FROM user_tools WHERE user_id = {user_id}")  # 50ms

    for tool in tools:
        request.tools_dict[tool.name] = tool

    return None

# Overhead: ~60ms
# Impact: 3% of total request time
# Verdict: ❌ Avoid! Pre-fetch and cache instead
```

### Use Case 5: External API Call (VERY SLOW ❌❌)

```python
def api_call(ctx, request):
    user_id = ctx.session.state.get("user_id")

    # ❌❌ VERY BAD: Blocks for network I/O
    config = requests.get(f"https://api.example.com/users/{user_id}/config").json()  # 500ms

    request.tools_dict.update(config["tools"])
    return None

# Overhead: ~520ms
# Impact: 26% of total request time
# Verdict: ❌❌ Unacceptable! This adds user-visible latency
```

---

## Best Practices for Minimizing Overhead

### ✅ DO: Use In-Memory State

```python
# FAST: State is already in memory
def fast_callback(ctx, request):
    tier = ctx.session.state.get("tier")  # Instant dict lookup

    if tier == "premium":
        request.tools_dict["premium_tool"] = premium_tool

    return None
```

### ✅ DO: Pre-Compute and Cache

```python
# Cache expensive computations
_tool_cache = {}

def cached_callback(ctx, request):
    user_id = ctx.session.state.get("user_id")

    # Check cache first
    if user_id not in _tool_cache:
        # Only compute once per user
        _tool_cache[user_id] = compute_user_tools(user_id)

    request.tools_dict.update(_tool_cache[user_id])
    return None
```

### ✅ DO: Keep Logic Simple

```python
# Simple conditional logic is fast
def simple_logic(ctx, request):
    # 3 state lookups: ~0.1ms
    tier = ctx.session.state.get("tier")
    role = ctx.session.state.get("role")
    features = ctx.session.state.get("features", [])

    # Simple if/else: ~0.01ms
    if tier == "premium" and role == "admin":
        request.tools_dict.update(admin_premium_tools)
    elif tier == "premium":
        request.tools_dict.update(premium_tools)
    else:
        request.tools_dict.update(basic_tools)

    return None
```

### ❌ DON'T: Make Network Calls

```python
# BAD: Adds 500ms+ latency
def bad_callback(ctx, request):
    user_id = ctx.session.state.get("user_id")

    # ❌ Network I/O blocks everything
    tools = requests.get(f"https://api.example.com/tools/{user_id}").json()

    request.tools_dict.update(tools)
    return None

# GOOD: Pre-fetch in a DummyAgent before LLM agent
def pre_fetch_tools(ctx):
    """DummyAgent that runs before LLM agent."""
    user_id = ctx.session.state.get("user_id")

    # Fetch once and store in state
    tools = requests.get(f"https://api.example.com/tools/{user_id}").json()
    ctx.session.state["user_tools"] = tools

    return "Tools loaded"

# Then callback just reads from state (fast!)
def fast_callback(ctx, request):
    tools = ctx.session.state.get("user_tools", {})
    request.tools_dict.update(tools)
    return None
```

### ❌ DON'T: Query Databases

```python
# BAD: Adds 50-100ms+ latency
def bad_callback(ctx, request):
    user_id = ctx.session.state.get("user_id")

    # ❌ Database query blocks
    tools = db.query("SELECT * FROM tools WHERE user_id = ?", user_id)

    request.tools_dict.update(tools)
    return None

# GOOD: Load tools at session creation
async def create_session_with_tools(user_id):
    # Query once when session is created
    tools = db.query("SELECT * FROM tools WHERE user_id = ?", user_id)

    await session_service.create_session(
        app_name="App",
        user_id=user_id,
        session_id=session_id,
        state={"user_tools": tools}  # Store in state
    )

# Callback just reads from state (instant!)
def fast_callback(ctx, request):
    tools = ctx.session.state.get("user_tools", {})
    request.tools_dict.update(tools)
    return None
```

### ❌ DON'T: Do Heavy Computation

```python
# BAD: Heavy computation
def bad_callback(ctx, request):
    data = ctx.session.state.get("large_dataset")

    # ❌ Complex computation blocks
    for item in data:
        result = expensive_algorithm(item)  # 100ms

    return None

# GOOD: Pre-compute in a DummyAgent
def pre_compute(ctx):
    """DummyAgent that runs before LLM agent."""
    data = ctx.session.state.get("large_dataset")

    # Do expensive work once
    results = [expensive_algorithm(item) for item in data]

    # Store results
    ctx.session.state["computed_results"] = results

    return "Computation complete"

# Callback just uses pre-computed results (fast!)
def fast_callback(ctx, request):
    results = ctx.session.state.get("computed_results")
    # Use results...
    return None
```

---

## Measuring Callback Overhead

### Add Timing to Your Callbacks

```python
import time

def instrumented_callback(ctx, request):
    start = time.perf_counter()

    # Your callback logic
    tier = ctx.session.state.get("tier")
    if tier == "premium":
        request.tools_dict.update(premium_tools)

    elapsed = (time.perf_counter() - start) * 1000

    # Log if it's slow
    if elapsed > 10:
        print(f"⚠️ Callback took {elapsed:.2f}ms (consider optimization)")

    # Store in state for monitoring
    ctx.session.state["_callback_time_ms"] = elapsed

    return None
```

### Profile in Production

```python
import time
from collections import defaultdict

# Global metrics
callback_times = defaultdict(list)

def monitored_callback(ctx, request):
    start = time.perf_counter()

    # Your logic
    tier = ctx.session.state.get("tier")
    request.tools_dict.update(get_tools_for_tier(tier))

    # Record metrics
    elapsed = (time.perf_counter() - start) * 1000
    callback_times["tool_loading"].append(elapsed)

    # Alert if p95 is high
    if len(callback_times["tool_loading"]) % 100 == 0:
        times = sorted(callback_times["tool_loading"])
        p95 = times[int(len(times) * 0.95)]

        if p95 > 50:
            print(f"🚨 P95 callback time: {p95:.2f}ms")

    return None
```

---

## When Overhead Matters

### Overhead is NEGLIGIBLE when:
- ✅ Simple state lookups (dict access)
- ✅ Conditional logic (if/else)
- ✅ Tool dictionary updates (<100 tools)
- ✅ String templating
- ✅ In-memory cache lookups

**These are all <5ms and don't impact user experience.**

### Overhead is NOTICEABLE when:
- ⚠️ Complex algorithm (>10ms)
- ⚠️ Large data processing (>50ms)
- ⚠️ Many nested loops (>20ms)
- ⚠️ Cache misses requiring computation (varies)

**Consider optimization if >10ms, but still usually acceptable.**

### Overhead is UNACCEPTABLE when:
- ❌ Database queries (50-200ms)
- ❌ Network/API calls (100-1000ms)
- ❌ File I/O (10-100ms)
- ❌ Synchronous external service calls (varies)

**These add user-visible latency and should be avoided!**

---

## Optimization Strategies

### 1. Pre-Fetch Data Before Agent Runs

```python
# Use a DummyAgent to pre-fetch expensive data
fetcher = DummyAgent(
    name="data_fetcher",
    logic_function=lambda ctx: fetch_expensive_data(ctx),
    output_key="expensive_data"
)

llm_agent = LlmAgent(
    name="llm",
    instruction="...",
    before_model_callback=lambda ctx, req: use_prefetched_data(ctx, req)
)

workflow = SequentialAgent(
    name="workflow",
    sub_agents=[fetcher, llm_agent]  # Fetch first, then use
)
```

### 2. Cache at Session Creation

```python
# Load everything once when session is created
async def create_enriched_session(user_id):
    # Query database once
    user_tools = await db.get_user_tools(user_id)
    user_config = await db.get_user_config(user_id)

    # Store in session state
    await session_service.create_session(
        app_name="App",
        user_id=user_id,
        session_id=session_id,
        state={
            "tools": user_tools,
            "config": user_config,
        }
    )

# Callbacks just read from state (instant!)
def callback(ctx, request):
    tools = ctx.session.state.get("tools")
    request.tools_dict.update(tools)
    return None
```

### 3. Use In-Memory Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user_tools(user_id: str):
    """Expensive computation, cached in memory."""
    # Only computed once per user_id
    return compute_tools(user_id)

def cached_callback(ctx, request):
    user_id = ctx.session.state.get("user_id")

    # Fast cache lookup
    tools = get_user_tools(user_id)
    request.tools_dict.update(tools)

    return None
```

### 4. Lazy Loading with Flags

```python
def lazy_callback(ctx, request):
    # Only load tools if user has accessed them before
    has_used_advanced = ctx.session.state.get("used_advanced_features", False)

    if has_used_advanced:
        # Load full toolset
        request.tools_dict.update(all_tools)
    else:
        # Just load basics (faster)
        request.tools_dict.update(basic_tools)

    return None
```

---

## Real-World Performance Example

### Scenario: Multi-Tenant SaaS with Dynamic Tools

```python
# Configuration
- Users: 10,000
- Avg tools per user: 15
- Callback complexity: Medium (permission checking + tool loading)
- Callback time: 3ms
- LLM call time: 2000ms

# Impact
- Total request time without callback: 2000ms
- Total request time with callback: 2003ms
- Overhead: 0.15%
- User experience: No perceptible difference

# Monthly costs (assuming 1M requests)
- LLM API calls: $1000
- Callback execution: $0 (negligible compute)
- Net impact: ~0%

# Verdict: ✅ Callback overhead is irrelevant
```

### Scenario: Callback with Database Query (Anti-Pattern)

```python
# Configuration
- DB query in callback: 75ms
- LLM call time: 2000ms

# Impact
- Total request time without callback: 2000ms
- Total request time with callback: 2075ms
- Overhead: 3.75%
- User experience: Noticeable delay on fast connections

# Solution: Pre-load at session creation
- Load tools once: 75ms (one-time cost)
- Subsequent callbacks: <1ms
- Net result: 75ms spent once instead of on every turn

# Verdict: ❌ Bad pattern, but easy to fix
```

---

## Summary

### Performance Impact

| Callback Type | Overhead | Impact | Verdict |
|---------------|----------|--------|---------|
| Simple state lookup | <1ms | 0.05% | ✅ Excellent |
| Tool dictionary updates | 1-5ms | 0.1-0.25% | ✅ Great |
| Complex logic | 5-10ms | 0.25-0.5% | ✅ Good |
| In-memory cache | 2-10ms | 0.1-0.5% | ✅ Acceptable |
| Database query | 50-200ms | 2.5-10% | ❌ Avoid |
| Network/API call | 100-1000ms | 5-50% | ❌❌ Never |

### Key Takeaways

1. **Callback overhead is negligible** for typical use cases (<5ms vs 2000ms LLM)
2. **Avoid I/O operations** in callbacks (DB, network, files)
3. **Pre-fetch expensive data** in DummyAgents or at session creation
4. **Use in-memory state** for instant lookups
5. **Monitor callback times** in production (>10ms warrants investigation)

### When to Worry

- ⚠️ If callback consistently takes >10ms, consider optimization
- ❌ If callback takes >50ms, definitely optimize
- ❌❌ If callback does I/O, refactor immediately

**Bottom line**: Use callbacks freely for logic, but keep them synchronous and in-memory!
