"""
Dummy Agent - Deterministic Node Implementation

This agent acts like a LangGraph node - it executes deterministic logic and tool calls
without LLM involvement, but fully integrates with ADK's event system, transfers,
and context management.

Use cases:
- Pre/post-processing steps in agent workflows
- Deterministic decision routing
- Tool orchestration without LLM reasoning
- Data validation and transformation nodes
"""

from typing import Any, AsyncGenerator, Callable, Dict, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.tools.base_tool import BaseTool
from google.genai import types


class DummyAgent(BaseAgent):
    """
    Deterministic agent that executes predefined logic without LLM calls.

    Acts as a "node" in the agent graph, similar to LangGraph nodes, where:
    - Logic is deterministic and controlled by code
    - Can execute tools programmatically
    - Can read/write session state
    - Can transfer to other agents
    - Emits proper ADK events

    This is useful for:
    - Data preprocessing/postprocessing
    - Conditional routing based on state
    - Deterministic tool execution
    - Validation steps
    - State transformations
    """

    # Type for the execution function
    logic_function: Callable[[InvocationContext], Any] = lambda ctx: None
    tools: list[BaseTool] = []
    output_key: Optional[str] = None
    transfer_to: Optional[str] = None  # Name of agent to transfer to after execution

    def __init__(
        self,
        name: str,
        description: str = "",
        logic_function: Optional[Callable[[InvocationContext], Any]] = None,
        tools: Optional[list[BaseTool]] = None,
        output_key: Optional[str] = None,
        transfer_to: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize a deterministic agent node.

        Args:
            name: Agent name (must be unique identifier)
            description: Description of what this node does
            logic_function: Function that takes InvocationContext and returns result
                           Can be sync or async. Return value saved to output_key.
            tools: List of tools this agent can call programmatically
            output_key: Key to store the result in session state
            transfer_to: Name of agent to transfer control to after execution
        """
        super().__init__(
            name=name,
            description=description,
            logic_function=logic_function or (lambda ctx: None),
            tools=tools or [],
            output_key=output_key,
            transfer_to=transfer_to,
            sub_agents=[],  # Dummy agents don't have sub-agents
            **kwargs,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Execute deterministic logic and emit appropriate events.

        Flow:
        1. Execute the logic function
        2. Save result to state if output_key is set
        3. Emit text response event
        4. Handle tool calls if logic returns tool requests
        5. Transfer to another agent if configured
        6. Mark completion
        """

        # Execute the logic function
        result = self.logic_function(ctx)

        # Handle async logic functions
        if hasattr(result, "__await__"):
            result = await result

        # Save result to session state if output_key is configured
        if self.output_key and result is not None:
            ctx.session.state[self.output_key] = result

        # Create response content
        response_text = self._format_response(result)

        # Emit a text response event
        if response_text:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model", parts=[types.Part(text=response_text)]
                ),
            )

        # Handle tool execution if result contains tool calls
        if isinstance(result, dict) and "tool_calls" in result:
            async for event in self._execute_tools(ctx, result["tool_calls"]):
                yield event

        # Handle transfer to another agent if configured
        if self.transfer_to:
            yield self._create_transfer_event(ctx)

        # Mark agent as complete
        if ctx.is_resumable:
            ctx.set_agent_state(self.name, end_of_agent=True)
            yield self._create_agent_state_event(ctx)

    def _format_response(self, result: Any) -> str:
        """
        Format the result as a text response.

        Args:
            result: The result from logic_function

        Returns:
            Formatted string response
        """
        if result is None:
            return ""

        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            # If it's a tool call dict, don't show it as text
            if "tool_calls" in result:
                return f"[{self.name}] Executing {len(result['tool_calls'])} tool(s)"
            # Otherwise format the dict
            return f"[{self.name}] Result: {result}"

        return f"[{self.name}] {str(result)}"

    async def _execute_tools(
        self, ctx: InvocationContext, tool_calls: list[dict]
    ) -> AsyncGenerator[Event, None]:
        """
        Execute tool calls programmatically and emit events.

        Args:
            ctx: Invocation context
            tool_calls: List of dicts with 'tool_name' and 'args' keys

        Yields:
            Events for tool calls and responses
        """
        from google.adk.tools.tool_context import ToolContext

        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            tool_args = tool_call.get("args", {})

            # Find the tool
            tool = self._find_tool(tool_name)
            if not tool:
                # Emit error event
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=f"Error: Tool '{tool_name}' not found")],
                    ),
                )
                continue

            # Create tool context
            tool_context = ToolContext(
                session_state=ctx.session.state,
                user_id=ctx.session.user_id,
            )

            try:
                # Execute the tool
                result = tool.execute(tool_args, tool_context)

                # Handle async tools
                if hasattr(result, "__await__"):
                    result = await result

                # Emit tool response event
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=f"Tool '{tool_name}' result: {result}")],
                    ),
                )

                # Optionally save result to state
                if "output_key" in tool_call:
                    ctx.session.state[tool_call["output_key"]] = result

            except Exception as e:
                # Emit error event
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                text=f"Error executing tool '{tool_name}': {str(e)}"
                            )
                        ],
                    ),
                )

    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _create_transfer_event(self, ctx: InvocationContext) -> Event:
        """
        Create an event that transfers control to another agent.

        Args:
            ctx: Invocation context

        Returns:
            Event with transfer action
        """
        from google.adk.events.event_actions import EventActions

        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Transferring to {self.transfer_to}")],
            ),
            actions=EventActions(transfer_to_agent=self.transfer_to),
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple data validation node
    def validate_input(ctx: InvocationContext) -> Dict[str, Any]:
        """Validate user input from state."""
        user_input = ctx.session.state.get("user_input", "")

        if len(user_input) < 10:
            return {"valid": False, "error": "Input too short (minimum 10 characters)"}

        if not user_input.strip():
            return {"valid": False, "error": "Input cannot be empty"}

        return {"valid": True, "message": "Input validated successfully"}

    validator_agent = DummyAgent(
        name="validator",
        description="Validates user input",
        logic_function=validate_input,
        output_key="validation_result",
    )

    # Example 2: Conditional routing node
    def route_by_category(ctx: InvocationContext) -> str:
        """Route to different agents based on category."""
        category = ctx.session.state.get("category", "").lower()

        routing = {
            "technical": "technical_support_agent",
            "billing": "billing_agent",
            "general": "general_agent",
        }

        target = routing.get(category, "general_agent")
        return f"Routing to {target}"

    router_agent = DummyAgent(
        name="router",
        description="Routes requests based on category",
        logic_function=route_by_category,
        transfer_to="technical_support_agent",  # Will transfer after execution
    )

    # Example 3: Data transformation node
    def transform_data(ctx: InvocationContext) -> Dict[str, Any]:
        """Transform and enrich data from state."""
        raw_data = ctx.session.state.get("raw_data", {})

        # Perform transformations
        first = raw_data.get("first_name", "")
        last = raw_data.get("last_name", "")
        email = raw_data.get("email", "")
        transformed = {
            "processed_at": "2024-01-01",  # In real use, use datetime.now()
            "user_id": raw_data.get("id"),
            "full_name": f"{first} {last}",
            "email_domain": email.split("@")[-1] if "@" in email else "",
        }

        return transformed

    transformer_agent = DummyAgent(
        name="transformer",
        description="Transforms raw data into structured format",
        logic_function=transform_data,
        output_key="transformed_data",
    )

    # Example 4: Deterministic tool execution node
    from google.adk.tools.function_tool import FunctionTool

    def fetch_user_data(user_id: str) -> Dict[str, Any]:
        """Mock function to fetch user data."""
        return {"user_id": user_id, "name": "John Doe", "email": "john@example.com"}

    user_tool = FunctionTool(func=fetch_user_data)

    def execute_fetch(ctx: InvocationContext) -> Dict[str, Any]:
        """Deterministically execute the user fetch tool."""
        user_id = ctx.session.state.get("user_id", "unknown")

        # Return tool call specification
        return {
            "tool_calls": [
                {
                    "tool_name": "fetch_user_data",
                    "args": {"user_id": user_id},
                    "output_key": "user_data",
                }
            ]
        }

    fetch_agent = DummyAgent(
        name="fetcher",
        description="Fetches user data deterministically",
        logic_function=execute_fetch,
        tools=[user_tool],
    )

    # Example 5: Async logic function
    async def async_processing(ctx: InvocationContext) -> str:
        """Example async logic function."""
        import asyncio

        # Simulate async operation
        await asyncio.sleep(0.1)

        count = ctx.session.state.get("count", 0)
        ctx.session.state["count"] = count + 1

        return f"Processed item {count + 1}"

    async_agent = DummyAgent(
        name="async_processor",
        description="Performs async processing",
        logic_function=async_processing,
        output_key="process_result",
    )

    # Example of using in a workflow:
    """
    from google.adk.agents import LlmAgent
    from google.adk.agents.workflow.sequential_agent import SequentialAgent

    # Create workflow: validate -> transform -> route -> process
    workflow = SequentialAgent(
        name="data_pipeline",
        agents=[
            validator_agent,      # Validate input
            transformer_agent,    # Transform data
            router_agent,         # Route to appropriate handler
            # ... other agents
        ]
    )

    # Or use in custom agent with conditional logic:
    class CustomWorkflow(BaseAgent):
        async def _run_async_impl(self, ctx):
            # Run validator
            async with Aclosing(validator_agent.run_async(ctx)) as agen:
                async for event in agen:
                    yield event

            # Check validation result
            validation = ctx.session.state.get("validation_result", {})

            if validation.get("valid"):
                # Continue with transformer
                async with Aclosing(transformer_agent.run_async(ctx)) as agen:
                    async for event in agen:
                        yield event
            else:
                # Handle error
                yield Event(...)
    """
