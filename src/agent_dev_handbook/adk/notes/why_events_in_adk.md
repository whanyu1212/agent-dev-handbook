# Why Agents Always Emit Events in ADK

## TL;DR

Events are the **fundamental communication unit** in ADK. They serve as:
1. **Conversation history** - Every interaction is recorded as an event
2. **State snapshots** - Capture what happened and when
3. **Streaming protocol** - Enable real-time UI updates
4. **Agent coordination** - Allow agents to see what other agents did
5. **Debugging trace** - Provide full audit trail of execution

**Think of events as the "append-only log" of your agent system.**

---

## The Event-Driven Architecture

### Session = Events + State

In ADK, a `Session` is composed of:

```python
class Session:
    id: str                    # Session identifier
    app_name: str              # Application name
    user_id: str               # User identifier
    state: dict[str, Any]      # Mutable state (key-value store)
    events: list[Event]        # Immutable conversation history
    last_update_time: float    # Timestamp
```

**Key insight**:
- `state` = **mutable workspace** for data passing
- `events` = **immutable history** of everything that happened

### What is an Event?

```python
class Event:
    id: str                    # Unique event ID
    invocation_id: str         # Which agent run this belongs to
    author: str                # 'user' or agent name
    content: Content           # The actual message/response
    actions: EventActions      # Actions taken (tool calls, transfers, etc.)
    branch: str                # Agent hierarchy path
    timestamp: float           # When this happened
```

An Event captures:
1. **Who** said/did something (author)
2. **What** they said/did (content + actions)
3. **When** it happened (timestamp)
4. **Where** in the agent tree (branch)
5. **Context** of execution (invocation_id)

---

## Why Events, Not Just Return Values?

### 1. **Conversation History (Memory)**

Events form the conversation history that enables:
- Context for future turns
- Multi-turn dialogue
- Agent memory across invocations

```python
# Without events (bad):
result = agent.run(input)  # Lost after return

# With events (ADK):
async for event in agent.run_async(ctx):
    # Event is stored in session.events
    # Available to all future agent calls
    yield event
```

**Example**: User asks "What did I ask about earlier?" - The agent can look back through events to find previous questions.

### 2. **Streaming & Real-Time Updates**

Events enable **streaming responses** to users as they're generated:

```python
# Frontend receives events in real-time
async for event in runner.run_async(user_id, session_id):
    if event.content:
        # Update UI immediately with partial response
        display_partial_text(event.content.parts[0].text)
```

**Without events**: User waits for entire response before seeing anything.
**With events**: User sees text streaming token-by-token (like ChatGPT).

### 3. **Multi-Agent Coordination**

Events let agents see what other agents did:

```python
async def _run_async_impl(self, ctx: InvocationContext):
    # Get conversation history
    events = ctx._get_events(current_invocation=True, current_branch=True)

    # See what the previous agent said
    last_event = events[-1]
    if last_event.author == "analyzer_agent":
        # React to what analyzer did
        analysis = last_event.content.parts[0].text
```

**Use case**:
- Critic agent reads story from generator agent's event
- Router agent checks which agent was called last
- Validator sees what tool calls were made

### 4. **Tool Calls & Responses**

Events capture the **full tool execution lifecycle**:

```python
# Event 1: Agent requests tool call
Event(
    author="my_agent",
    content=Content(parts=[Part(function_call=FunctionCall(...))])
)

# Event 2: Tool response
Event(
    author="user",  # Tool responses come as user events
    content=Content(parts=[Part(function_response=FunctionResponse(...))])
)

# Event 3: Agent processes tool result
Event(
    author="my_agent",
    content=Content(parts=[Part(text="Based on the data...")])
)
```

This enables:
- Resuming after tool execution
- Seeing tool call history
- Debugging tool interactions

### 5. **Agent Transfers**

Events communicate agent transfers:

```python
# Transfer event
Event(
    author="router_agent",
    content=Content(parts=[Part(text="Transferring to billing agent")]),
    actions=EventActions(transfer_to_agent="billing_agent")
)
```

The next agent can:
- See who transferred to them
- Understand the context of the transfer
- Access the full conversation leading up to the transfer

### 6. **Debugging & Observability**

Events provide a **complete audit trail**:

```python
for event in session.events:
    print(f"{event.timestamp} | {event.author} | {event.content}")
```

Output:
```
1704096000.0 | user | I need help with billing
1704096001.2 | router | Routing to billing agent
1704096001.5 | billing_agent | Let me check your account
1704096002.1 | billing_agent | [tool: fetch_billing_info]
1704096003.0 | user | [tool response: {...}]
1704096003.5 | billing_agent | Your balance is $50
```

This shows:
- Exact execution flow
- Agent decision points
- Tool calls and results
- Timing information

### 7. **Resumability & Pause/Resume**

Events enable pausing and resuming agent execution:

```python
# Check if we need to resume from a previous state
agent_state = self._load_agent_state(ctx, BaseAgentState)

if agent_state is not None:
    # Resume from where we left off
    # Events tell us what already happened
    events = ctx._get_events(current_invocation=True, current_branch=True)
    last_event = events[-1]

    if last_event.author == "sub_agent":
        # Continue from sub_agent completion
        ...
```

**Use case**: Long-running tool calls that need to wait for external processing.

---

## Event Flow in Practice

### Simple Example: User Message to Agent Response

```python
# User sends message -> Creates Event
user_event = Event(
    author="user",
    content=Content(parts=[Part(text="What's the weather?")])
)
session.events.append(user_event)

# Agent processes and emits Event
async def _run_async_impl(self, ctx):
    # Agent generates response
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=Content(parts=[Part(text="Let me check...")])
    )

    # Agent calls weather tool
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=Content(parts=[Part(function_call=FunctionCall(...))])
    )

    # Tool result comes back as Event (handled by framework)
    # ...

    # Agent provides final answer
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=Content(parts=[Part(text="It's sunny, 72°F")])
    )
```

### Complex Example: Multi-Agent Workflow

```python
# Router agent emits decision
yield Event(
    author="router",
    content=Content(parts=[Part(text="Detected technical question")]),
    actions=EventActions(transfer_to_agent="tech_support")
)

# Tech support agent sees the events
async def _run_async_impl(self, ctx):
    # Can access all previous events
    events = ctx._get_events(current_invocation=True, current_branch=True)

    # See original user question
    user_question = events[0].content.parts[0].text

    # See router's reasoning
    router_decision = events[1].content.parts[0].text

    # Respond accordingly
    yield Event(
        author="tech_support",
        content=Content(parts=[Part(text=f"I see you need help with: {user_question}")])
    )
```

---

## Events vs State: When to Use Which?

### Use **Events** for:
✅ Conversation messages (user input, agent responses)
✅ Tool calls and responses
✅ Agent transfers
✅ Actions taken (what happened)
✅ Anything you want in conversation history
✅ Debugging traces

### Use **State** for:
✅ Intermediate data passing between agents
✅ Temporary computation results
✅ Configuration values
✅ Counters, flags, metadata
✅ Data you DON'T want shown to user

**Example**:

```python
# Event - User sees this in conversation
yield Event(
    author="analyzer",
    content=Content(parts=[Part(text="Analysis complete")])
)

# State - Internal data passing
ctx.session.state["analysis_result"] = {
    "sentiment": "positive",
    "confidence": 0.92,
    "keywords": ["happy", "satisfied"]
}

# Next agent reads from state
sentiment = ctx.session.state["analysis_result"]["sentiment"]
```

---

## Why Yield Events Instead of Collect and Return?

### ❌ **Bad** (Collecting events):
```python
async def _run_async_impl(self, ctx):
    events = []

    # Generate all events first
    events.append(Event(...))
    events.append(Event(...))
    events.append(Event(...))

    # Return at end
    for event in events:
        yield event
```

**Problems**:
- User sees nothing until complete
- No streaming
- Wastes memory
- Can't cancel mid-execution

### ✅ **Good** (Streaming events):
```python
async def _run_async_impl(self, ctx):
    # Yield immediately as generated
    yield Event(...)  # User sees this RIGHT NOW

    # Do some work
    result = await some_operation()

    yield Event(...)  # User sees this as soon as ready
```

**Benefits**:
- Real-time feedback
- Progressive rendering
- Memory efficient
- Cancellable

---

## Event Lifecycle

1. **Creation**: Agent creates Event object
2. **Yielding**: Agent yields event (becomes available to consumers)
3. **Storage**: Framework appends to `session.events`
4. **Streaming**: Event sent to client/UI in real-time
5. **Persistence**: Session saved to database with all events
6. **Retrieval**: Future agent runs can access via `ctx._get_events()`

---

## Practical Implications for Custom Agents

### You MUST Yield Events Because:

1. **Framework expects it**: The `_run_async_impl` signature is `AsyncGenerator[Event, None]`
2. **Session history**: Events are automatically stored in session
3. **UI rendering**: Frontend expects events to display conversation
4. **Agent coordination**: Other agents need to see your output
5. **Observability**: Logging/monitoring systems consume events

### Minimal Valid Agent:

```python
async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    # Must yield at least one event!
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        content=Content(parts=[Part(text="I did something")])
    )
```

### Even "Silent" Nodes Must Emit Events:

```python
# Even if just updating state, emit an event
async def _run_async_impl(self, ctx):
    # Update state silently
    ctx.session.state["processed"] = True

    # Still emit event (can be minimal)
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        content=Content(parts=[Part(text="[Processing complete]")])
    )

    # Mark completion
    if ctx.is_resumable:
        ctx.set_agent_state(self.name, end_of_agent=True)
        yield self._create_agent_state_event(ctx)
```

---

## Summary: Events Are The "Source of Truth"

In ADK's architecture:

```
Events = Immutable History (what happened)
State = Mutable Workspace (current data)
```

**Events answer**:
- What was said?
- What actions were taken?
- Who did what when?
- How did we get here?

**State answers**:
- What's the current value of X?
- What should I pass to the next agent?
- What configuration is active?

Both are essential, but **events are the foundation** because:
1. They enable conversation memory
2. They enable streaming UX
3. They enable multi-agent coordination
4. They enable debugging and observability
5. They enable pause/resume functionality

**Bottom line**: If it happened, it should be an event. If it's just data, it can be state.
