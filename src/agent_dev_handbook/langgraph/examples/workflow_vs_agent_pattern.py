"""
LangGraph: Workflow vs Agent Pattern

This example demonstrates the fundamental distinction between:
1. **Workflow Pattern**: Deterministic, predefined control flow
2. **Agent Pattern**: Dynamic, LLM-driven decision making

Key Differences:
- Workflows use conditional edges based on data/rules
- Agents use LLMs to decide next actions dynamically
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages

# =============================================================================
# WORKFLOW PATTERN: Deterministic Flow
# =============================================================================
# Characteristics:
# - Control flow determined by rules/conditions
# - No LLM involvement in routing decisions
# - Predictable, reproducible execution paths
# - Good for: ETL pipelines, data processing, validation flows


class WorkflowState(TypedDict):
    """State for deterministic workflow."""

    user_input: str
    is_valid: bool
    category: str
    result: str
    error: str | None


def validate_input(state: WorkflowState) -> dict:
    """Validate user input using deterministic rules."""
    user_input = state["user_input"]

    # Simple validation logic
    if not user_input or len(user_input.strip()) < 3:
        return {"is_valid": False, "error": "Input too short"}

    return {"is_valid": True, "error": None}


def categorize_input(state: WorkflowState) -> dict:
    """Categorize input based on keywords (deterministic)."""
    user_input = state["user_input"].lower()

    # Rule-based categorization
    if any(word in user_input for word in ["code", "program", "function", "bug"]):
        category = "technical"
    elif any(word in user_input for word in ["weather", "forecast", "temperature"]):
        category = "weather"
    elif any(word in user_input for word in ["news", "article", "story"]):
        category = "news"
    else:
        category = "general"

    return {"category": category}


def process_technical(state: WorkflowState) -> dict:
    """Process technical queries."""
    return {"result": f"Technical processing: {state['user_input']}"}


def process_weather(state: WorkflowState) -> dict:
    """Process weather queries."""
    return {"result": f"Weather lookup: {state['user_input']}"}


def process_news(state: WorkflowState) -> dict:
    """Process news queries."""
    return {"result": f"News search: {state['user_input']}"}


def process_general(state: WorkflowState) -> dict:
    """Process general queries."""
    return {"result": f"General response: {state['user_input']}"}


def handle_error(state: WorkflowState) -> dict:
    """Handle validation errors."""
    return {"result": f"Error: {state['error']}"}


def route_after_validation(state: WorkflowState) -> Literal["categorize", "error"]:
    """Deterministic routing based on validation result."""
    if state["is_valid"]:
        return "categorize"
    return "error"


def route_by_category(
    state: WorkflowState,
) -> Literal["technical", "weather", "news", "general"]:
    """Deterministic routing based on category."""
    return state["category"]  # type: ignore


def build_workflow_graph() -> StateGraph:
    """
    Build a deterministic workflow graph.

    Flow:
    1. Validate input (rule-based)
    2. If invalid -> error handler
    3. If valid -> categorize (keyword-based)
    4. Route to appropriate processor based on category
    """
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("validate", validate_input)
    workflow.add_node("categorize", categorize_input)
    workflow.add_node("error", handle_error)
    workflow.add_node("technical", process_technical)
    workflow.add_node("weather", process_weather)
    workflow.add_node("news", process_news)
    workflow.add_node("general", process_general)

    # Entry point
    workflow.set_entry_point("validate")

    # Conditional routing after validation
    workflow.add_conditional_edge(
        "validate",
        route_after_validation,
        {"categorize": "categorize", "error": "error"},
    )

    # Conditional routing after categorization
    workflow.add_conditional_edge(
        "categorize",
        route_by_category,
        {
            "technical": "technical",
            "weather": "weather",
            "news": "news",
            "general": "general",
        },
    )

    # All processing nodes end
    workflow.add_edge("error", END)
    workflow.add_edge("technical", END)
    workflow.add_edge("weather", END)
    workflow.add_edge("news", END)
    workflow.add_edge("general", END)

    return workflow


# =============================================================================
# AGENT PATTERN: LLM-Driven Dynamic Flow
# =============================================================================
# Characteristics:
# - LLM decides next actions based on context
# - Dynamic tool selection and routing
# - Adapts to unexpected inputs
# - Good for: Conversational AI, research tasks, complex reasoning


class AgentState(TypedDict):
    """State for agent pattern with message history."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_action: str  # LLM decides this


# Mock tools for demonstration
def search_tool(query: str) -> str:
    """Mock search tool."""
    return f"Search results for: {query}"


def calculator_tool(expression: str) -> str:
    """Mock calculator tool."""
    try:
        result = eval(expression)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def code_tool(code: str) -> str:
    """Mock code execution tool."""
    return f"Executed code: {code[:50]}..."


TOOLS_MAP = {
    "search": search_tool,
    "calculator": calculator_tool,
    "code": code_tool,
}


def agent_node(state: AgentState) -> dict:
    """
    Agent node: LLM decides what to do next.

    Unlike workflow pattern, the LLM:
    - Analyzes user intent dynamically
    - Chooses appropriate tools
    - Determines if more steps needed
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # System prompt guides LLM decision making
    system_prompt = """You are a helpful assistant with access to tools:
- search: Look up information
- calculator: Perform calculations
- code: Execute code snippets

Analyze the user's message and decide:
1. Which tool to use (or none if you can answer directly)
2. Whether you need multiple steps
3. When you have enough info to provide final answer

Respond with:
- If using tool: "TOOL: <tool_name>: <input>"
- If done: "FINAL: <answer>"
"""

    messages = [
        HumanMessage(content=system_prompt),
        *state["messages"],
    ]

    response = llm.invoke(messages)

    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute tools based on agent decision."""
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return {"messages": []}

    content = last_message.content

    # Parse tool call from agent response
    if isinstance(content, str) and content.startswith("TOOL:"):
        parts = content.split(":", 2)
        if len(parts) >= 3:
            tool_name = parts[1].strip()
            tool_input = parts[2].strip()

            if tool_name in TOOLS_MAP:
                result = TOOLS_MAP[tool_name](tool_input)
                return {"messages": [HumanMessage(content=f"Tool result: {result}")]}

    return {"messages": []}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Router function: LLM-driven decision.

    Unlike workflow pattern where we use simple rules,
    here we check if the LLM decided to use a tool or finish.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        content = last_message.content
        if isinstance(content, str):
            if content.startswith("TOOL:"):
                return "tools"
            if content.startswith("FINAL:"):
                return "end"

    # Default: continue conversation
    return "tools"


def build_agent_graph() -> StateGraph:
    """
    Build an agent-driven graph.

    Flow:
    1. Agent analyzes message and decides action (LLM-driven)
    2. If agent calls tool -> execute tool -> back to agent
    3. If agent provides final answer -> end
    4. Loop continues until agent decides to finish
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Entry point
    graph.set_entry_point("agent")

    # Agent decides whether to use tools or finish
    graph.add_conditional_edge(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # After tools, always return to agent for next decision
    graph.add_edge("tools", "agent")

    return graph


# =============================================================================
# COMPARISON RUNNER
# =============================================================================


def run_workflow_example():
    """Run workflow pattern example."""
    print("\n" + "=" * 70)
    print("WORKFLOW PATTERN: Deterministic Flow")
    print("=" * 70)

    workflow = build_workflow_graph()
    app = workflow.compile()

    test_inputs = [
        "How to fix a Python bug?",  # -> technical
        "What's the weather like?",  # -> weather
        "Latest tech news",  # -> news
        "Hi",  # -> too short, error
        "Tell me a joke",  # -> general
    ]

    for user_input in test_inputs:
        print(f"\nInput: '{user_input}'")
        print("-" * 70)

        result = app.invoke({"user_input": user_input})

        print(f"Valid: {result.get('is_valid', 'N/A')}")
        print(f"Category: {result.get('category', 'N/A')}")
        print(f"Result: {result.get('result', 'N/A')}")
        print(f"Error: {result.get('error', 'N/A')}")


def run_agent_example():
    """Run agent pattern example."""
    print("\n" + "=" * 70)
    print("AGENT PATTERN: LLM-Driven Dynamic Flow")
    print("=" * 70)
    print("\nNOTE: This requires OPENAI_API_KEY environment variable")
    print("The agent dynamically decides which tools to use\n")

    agent = build_agent_graph()
    app = agent.compile()

    test_queries = [
        "What is 25 * 34?",  # Should use calculator
        "Search for Python tutorials",  # Should use search
        "What is the capital of France?",  # Might answer directly or search
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)

        try:
            # Stream to see agent's reasoning process
            for state in app.stream({"messages": [HumanMessage(content=query)]}):
                for node, output in state.items():
                    if "messages" in output:
                        for msg in output["messages"]:
                            print(f"[{node}] {msg.content}")
        except Exception as e:
            print(f"Error (likely missing API key): {e}")
            print("Set OPENAI_API_KEY to run agent pattern example")
            break


def main():
    """Run both examples to demonstrate differences."""
    print("\n" + "=" * 70)
    print("LangGraph: Workflow vs Agent Pattern Comparison")
    print("=" * 70)

    print("\n📋 KEY DIFFERENCES:")
    print("-" * 70)
    print("WORKFLOW PATTERN:")
    print("  ✓ Deterministic routing based on rules")
    print("  ✓ No LLM calls for decision making")
    print("  ✓ Predictable, fast execution")
    print("  ✓ Use cases: Data pipelines, validation, routing")
    print()
    print("AGENT PATTERN:")
    print("  ✓ LLM decides next actions dynamically")
    print("  ✓ Can handle unexpected inputs gracefully")
    print("  ✓ Adapts strategy based on context")
    print("  ✓ Use cases: Conversational AI, research, complex reasoning")
    print("=" * 70)

    # Run workflow example (no API key needed)
    run_workflow_example()

    # Run agent example (requires OpenAI API key)
    run_agent_example()


if __name__ == "__main__":
    main()
