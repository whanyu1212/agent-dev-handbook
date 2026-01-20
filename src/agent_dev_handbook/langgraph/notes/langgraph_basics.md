# LangGraph Basics

## Overview

LangGraph is a library for building stateful, graph-based agents with cyclic workflows. Unlike traditional chain-based approaches (like LangChain's LCEL), LangGraph enables you to create agents that can loop, branch, and maintain complex state across multiple steps.

**Key Features:**
- **Stateful graphs**: Maintain and update state across multiple steps
- **Cyclic workflows**: Support loops and conditional branching
- **Human-in-the-loop**: Built-in support for human approval/intervention
- **Persistence**: Save and resume workflows at any point
- **Streaming**: Stream outputs, tokens, and state updates in real-time

## Core Concepts

### 1. StateGraph

The `StateGraph` is the fundamental building block in LangGraph. It defines:
- The **state schema** (what data flows through the graph)
- The **nodes** (individual processing steps)
- The **edges** (how nodes connect)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state schema
class AgentState(TypedDict):
    messages: list[str]
    next_step: str
    iteration: int

# Create graph
workflow = StateGraph(AgentState)
```

### 2. State

State in LangGraph is a dictionary-like object that flows through the graph. Each node receives the current state and returns updates.

**Key characteristics:**
- **Immutable by default**: Each node creates a new state version
- **Typed**: Define schema using TypedDict or Pydantic models
- **Reducers**: Control how state updates merge (append, overwrite, custom logic)

```python
from typing import Annotated
from operator import add

class AgentState(TypedDict):
    # Default: overwrite previous value
    counter: int

    # Custom reducer: append to list
    messages: Annotated[list[str], add]

    # Explicit overwrite
    status: str
```

**Common Reducers:**
- `operator.add` - Concatenate lists/strings, sum numbers
- Custom functions - Define your own merge logic

### 3. Nodes

Nodes are Python functions that process state. Each node:
- Receives the current state as input
- Returns a dictionary of state updates
- Can be sync or async functions

```python
def process_input(state: AgentState) -> dict:
    """Node that processes user input."""
    messages = state["messages"]
    new_message = f"Processed: {messages[-1]}"

    # Return state updates
    return {
        "messages": [new_message],
        "iteration": state["iteration"] + 1
    }

# Add node to graph
workflow.add_node("process", process_input)
```

**Node Function Signature:**
```python
def node_function(state: StateType) -> dict | StateType:
    # Process state
    return {"key": "updated_value"}
```

### 4. Edges

Edges define the flow between nodes. LangGraph supports three types:

#### Simple Edges
Direct connection from one node to another:
```python
workflow.add_edge("node_a", "node_b")
```

#### Conditional Edges
Dynamic routing based on state:
```python
def route_decision(state: AgentState) -> str:
    """Decide which node to go to next."""
    if state["iteration"] > 5:
        return "finish"
    else:
        return "continue"

workflow.add_conditional_edge(
    "process",
    route_decision,
    {
        "finish": END,
        "continue": "process"
    }
)
```

#### Entry and End Points
- **Entry point**: `workflow.set_entry_point("start_node")`
- **End point**: Use `END` constant or `workflow.set_finish_point("end_node")`

### 5. Compilation

After defining the graph structure, compile it into a runnable:

```python
app = workflow.compile()

# Run the graph
result = app.invoke({"messages": ["Hello"], "iteration": 0})

# Stream results
for chunk in app.stream({"messages": ["Hello"], "iteration": 0}):
    print(chunk)
```

## Basic Workflow Pattern

Here's a complete minimal example:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define State
class AgentState(TypedDict):
    input: str
    output: str
    steps: int

# 2. Create Graph
workflow = StateGraph(AgentState)

# 3. Define Nodes
def process(state: AgentState) -> dict:
    return {
        "output": f"Processed: {state['input']}",
        "steps": state.get("steps", 0) + 1
    }

def check_done(state: AgentState) -> str:
    """Conditional edge function."""
    if state["steps"] >= 3:
        return "end"
    return "continue"

# 4. Build Graph
workflow.add_node("process", process)
workflow.set_entry_point("process")

workflow.add_conditional_edge(
    "process",
    check_done,
    {
        "end": END,
        "continue": "process"
    }
)

# 5. Compile and Run
app = workflow.compile()
result = app.invoke({"input": "Hello", "steps": 0})
```

## State Reducers in Detail

Reducers control how state updates merge when multiple updates target the same key.

### Default Behavior (Overwrite)
```python
class State(TypedDict):
    value: int  # Latest value overwrites

# Node returns {"value": 5}
# State becomes {"value": 5}
```

### Append Behavior
```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list[str], add]

# Node 1 returns {"messages": ["msg1"]}
# Node 2 returns {"messages": ["msg2"]}
# Final state: {"messages": ["msg1", "msg2"]}
```

### Custom Reducer
```python
def merge_dicts(left: dict, right: dict) -> dict:
    """Custom reducer that merges dictionaries."""
    return {**left, **right}

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

## Common Patterns

### 1. Agent Loop Pattern
```python
def should_continue(state: AgentState) -> str:
    """Check if agent should continue or finish."""
    if "FINAL ANSWER" in state["messages"][-1]:
        return "end"
    return "continue"

workflow.add_conditional_edge(
    "agent",
    should_continue,
    {
        "continue": "agent",  # Loop back
        "end": END
    }
)
```

### 2. Tool Calling Pattern
```python
def call_tools(state: AgentState) -> dict:
    """Execute tools based on agent decision."""
    tool_calls = state.get("tool_calls", [])
    results = []

    for tool_call in tool_calls:
        result = execute_tool(tool_call)
        results.append(result)

    return {"tool_results": results}

workflow.add_node("agent", agent_node)
workflow.add_node("tools", call_tools)

workflow.add_conditional_edge(
    "agent",
    lambda s: "tools" if s.get("tool_calls") else "end",
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")  # Loop back to agent
```

### 3. Multi-Agent Collaboration
```python
def router(state: AgentState) -> str:
    """Route to appropriate specialist agent."""
    query_type = classify_query(state["query"])

    if query_type == "code":
        return "code_agent"
    elif query_type == "research":
        return "research_agent"
    else:
        return "general_agent"

workflow.add_node("router", router_node)
workflow.add_node("code_agent", code_agent_node)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("general_agent", general_agent_node)

workflow.add_conditional_edge(
    "router",
    router,
    {
        "code_agent": "code_agent",
        "research_agent": "research_agent",
        "general_agent": "general_agent"
    }
)

# All agents end at same place
for agent in ["code_agent", "research_agent", "general_agent"]:
    workflow.add_edge(agent, END)
```

## Execution Modes

### 1. Invoke (Blocking)
Run the entire graph and return final state:
```python
result = app.invoke(initial_state)
print(result)  # Final state
```

### 2. Stream (Progressive Updates)
Get state updates as they occur:
```python
for state in app.stream(initial_state):
    print(state)  # Intermediate states
```

### 3. Stream with Mode
Control what gets streamed:
```python
# Stream state updates only
for chunk in app.stream(initial_state, stream_mode="updates"):
    print(chunk)

# Stream individual values (for debugging)
for chunk in app.stream(initial_state, stream_mode="values"):
    print(chunk)

# Stream both
for chunk in app.stream(initial_state, stream_mode=["updates", "values"]):
    print(chunk)
```

## Error Handling

### Node-Level Error Handling
```python
def safe_node(state: AgentState) -> dict:
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

### Graph-Level Error Handling
```python
try:
    result = app.invoke(initial_state)
except Exception as e:
    print(f"Graph execution failed: {e}")
```

## Best Practices

### 1. State Design
- **Keep state flat**: Avoid deep nesting when possible
- **Use TypedDict or Pydantic**: Type safety prevents bugs
- **Choose appropriate reducers**: Match merge behavior to data semantics
- **Minimize state size**: Only include necessary data

### 2. Node Design
- **Single responsibility**: Each node should do one thing well
- **Pure functions when possible**: Easier to test and reason about
- **Clear return types**: Always return dict with explicit keys
- **Handle edge cases**: Validate inputs and handle missing data

### 3. Graph Structure
- **Start simple**: Begin with linear flow, add complexity as needed
- **Use conditional edges sparingly**: Too many branches are hard to debug
- **Name nodes descriptively**: "process_input" not "node1"
- **Document routing logic**: Explain why conditional edges route the way they do

### 4. Debugging
- **Use stream mode**: See intermediate states
- **Add logging nodes**: Create nodes just for logging state
- **Visualize graph**: Use LangGraph's visualization tools
- **Test nodes independently**: Unit test node functions

## LangGraph vs Traditional Chains

| Feature | LangGraph | Traditional Chains |
|---------|-----------|-------------------|
| **Control Flow** | Cyclic, conditional | Linear, sequential |
| **State** | Explicit, mutable | Implicit, passed through |
| **Loops** | Native support | Difficult/impossible |
| **Branching** | Conditional edges | Limited |
| **Debugging** | Stream intermediate states | Harder to inspect |
| **Complexity** | Handles complex workflows | Best for simple flows |

## Next Steps

- **Checkpointing**: Learn to save/resume graph execution
- **Human-in-the-loop**: Add approval steps
- **Persistence**: Use MemorySaver or custom checkpointers
- **Subgraphs**: Compose graphs from smaller graphs
- **Tool Integration**: Connect LangChain tools to LangGraph nodes
- **Streaming**: Master token-level and state-level streaming

## Common Gotchas

1. **Forgetting to compile**: Must call `workflow.compile()` before running
2. **State reducer confusion**: Remember default is overwrite, not append
3. **Conditional edge return values**: Must match keys in edge mapping
4. **Missing END**: Graphs must have at least one path to END
5. **Node return format**: Always return dict, not full state object
6. **Async mixing**: Don't mix sync and async nodes without proper handling

## References

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **API Reference**: https://langchain-ai.github.io/langgraph/reference/graphs/
- **Tutorials**: https://langchain-ai.github.io/langgraph/tutorials/
- **Examples**: See `src/agent_dev_handbook/langgraph/examples/`
