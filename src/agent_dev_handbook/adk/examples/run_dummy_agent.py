"""
Dummy Agent Example with Callback Inspection

This demonstrates:
1. A 3-agent workflow: LlmAgent -> DummyAgent -> LlmAgent
2. Callback inspection showing what happens before/after each LLM call
3. How DummyAgent (no LLM) fits between LLM agents
4. State passing between agents

Run with: uv run python src/agent_dev_handbook/adk/examples/run_dummy_agent.py
"""

import asyncio
from pathlib import Path

from callback_inspector import (
    CallbackInspectorConfig,
    console,
    create_after_model_inspector,
    create_before_model_inspector,
    print_state_table,
    print_summary,
    print_workflow_header,
)
from dotenv import load_dotenv
from dummy_agent import DummyAgent
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Load environment
load_dotenv(Path(__file__).parent / ".env")


# Track statistics across the workflow
workflow_stats = {
    "llm_calls": 0,
    "total_prompt_tokens": 0,
    "total_response_tokens": 0,
    "agents_executed": [],
}


def stats_handler(inspection):
    """Custom handler to track statistics."""
    if inspection.callback_type == "after_model":
        workflow_stats["llm_calls"] += 1
        workflow_stats["agents_executed"].append(inspection.agent_name)

        usage = inspection.details.get("usage")
        if usage:
            workflow_stats["total_prompt_tokens"] += (
                usage.get("prompt_token_count") or 0
            )
            workflow_stats["total_response_tokens"] += (
                usage.get("candidates_token_count") or 0
            )


async def main():
    # Print header
    print_workflow_header(
        "DUMMY AGENT WORKFLOW WITH CALLBACK INSPECTION",
        "Workflow: generator (LLM) → doubler (Dummy) → interpreter (LLM)",
    )

    # Create inspector config with custom handler
    inspector_config = CallbackInspectorConfig(
        verbose=True,
        log_state=True,
        log_tools=True,
        log_contents=True,
        log_usage=True,
        log_methods=False,  # Set to True to see available methods
        custom_handler=stats_handler,
    )

    # Create inspectors
    before_inspector = create_before_model_inspector(inspector_config)
    after_inspector = create_after_model_inspector(inspector_config)

    # ========================================================================
    # Define the workflow: LlmAgent -> DummyAgent -> LlmAgent
    # ========================================================================

    # 1. LLM Agent: Generates a number (with callback inspection)
    generator_instruction = (
        "Generate a random number between 1 and 100. "
        "Just respond with the number only, nothing else."
    )
    generator = LlmAgent(
        name="generator",
        instruction=generator_instruction,
        output_key="generated_number",
        before_model_callback=before_inspector,
        after_model_callback=after_inspector,
    )

    # 2. DUMMY Agent: Doubles the number (NO LLM CALL - no callbacks needed)
    def double_number(ctx):
        """Deterministic logic - no LLM involved."""
        num_str = ctx.session.state.get("generated_number", "0")

        # Print a nice separator for the dummy agent
        console.print()
        console.rule(
            "[bold magenta]DUMMY AGENT EXECUTION[/bold magenta]", style="magenta"
        )

        # Show what we received
        input_tree = Tree("[bold yellow]Input from previous agent")
        input_tree.add(f"[cyan]generated_number[/cyan]: [white]{num_str!r}[/white]")
        console.print(input_tree)

        # Extract number from LLM response
        try:
            num = int("".join(filter(str.isdigit, str(num_str))))
        except ValueError:
            num = 0

        doubled = num * 2

        # Write to state for next agent
        ctx.session.state["doubled_number"] = doubled
        ctx.session.state["original_number"] = num

        # Show the computation
        computation_panel = Panel(
            f"[cyan]{num}[/cyan] × [yellow]2[/yellow] = [green bold]{doubled}[/green]",
            title="Computation (No LLM)",
            border_style="green",
        )
        console.print(computation_panel)

        # Show output state
        output_tree = Tree("[bold green]Output to session state")
        output_tree.add(f"[cyan]doubled_number[/cyan]: [white]{doubled}[/white]")
        output_tree.add(f"[cyan]original_number[/cyan]: [white]{num}[/white]")
        console.print(output_tree)

        console.rule(style="magenta dim")

        # Return message that goes into event
        return f"I doubled {num} to get {doubled}"

    doubler = DummyAgent(
        name="doubler",
        description="Doubles a number without using LLM",
        logic_function=double_number,
        output_key="doubler_result",
    )

    # 3. LLM Agent: Interprets the result (with callback inspection)
    interpreter = LlmAgent(
        name="interpreter",
        instruction=(
            "The original number was {original_number}. "
            "It was doubled to {doubled_number}. "
            "Explain this calculation in a friendly, brief way (1-2 sentences)."
        ),
        output_key="final_response",
        before_model_callback=before_inspector,
        after_model_callback=after_inspector,
    )

    # ========================================================================
    # Create Sequential Workflow
    # ========================================================================

    workflow = SequentialAgent(
        name="number_workflow",
        sub_agents=[generator, doubler, interpreter],
    )

    # ========================================================================
    # Setup and Run
    # ========================================================================

    console.print()
    console.print(Panel("[bold]Setting up session...[/bold]", border_style="blue"))

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="DummyTest",
        user_id="user1",
        session_id="session1",
        state={"workflow_name": "number_doubler"},
    )

    # Show initial state
    print_state_table(dict(session.state), "Initial Session State")

    runner = Runner(
        app_name="DummyTest", agent=workflow, session_service=session_service
    )

    console.print()
    console.print(
        Panel(
            "[bold green]Starting workflow execution...[/bold green]\n"
            "[dim]User message: 'Start the workflow'[/dim]",
            border_style="green",
        )
    )

    events = runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=types.Content(
            role="user", parts=[types.Part(text="Start the workflow")]
        ),
    )

    # ========================================================================
    # Process Events
    # ========================================================================

    console.print()
    console.rule("[bold]WORKFLOW EVENTS[/bold]", style="bright_blue")

    event_table = Table(title="Events Emitted", show_lines=True)
    event_table.add_column("Author", style="cyan", width=15)
    event_table.add_column("Content", style="white", max_width=60)

    async for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    text = (
                        part.text[:100] + "..." if len(part.text) > 100 else part.text
                    )
                    event_table.add_row(event.author, text)

    console.print(event_table)

    # ========================================================================
    # Check Final State
    # ========================================================================

    final_session = await session_service.get_session(
        app_name="DummyTest", user_id="user1", session_id="session1"
    )

    console.print()
    print_state_table(dict(final_session.state), "Final Session State")

    # ========================================================================
    # Summary
    # ========================================================================

    workflow_stats["agents_executed"] = " → ".join(workflow_stats["agents_executed"])

    print_summary(
        {
            "Total LLM Calls": workflow_stats["llm_calls"],
            "Total Prompt Tokens": workflow_stats["total_prompt_tokens"],
            "Total Response Tokens": workflow_stats["total_response_tokens"],
            "Execution Order": workflow_stats["agents_executed"],
        }
    )

    # Key takeaways
    console.print()
    console.print(
        Panel(
            "[bold green]KEY OBSERVATIONS[/bold green]\n\n"
            "1. [cyan]generator[/cyan] and [cyan]interpreter[/cyan] triggered "
            "callbacks (they use LLM)\n\n"
            "2. [magenta]doubler[/magenta] (DummyAgent) did NOT trigger callbacks\n"
            "   (no LLM call - pure deterministic logic)\n\n"
            "3. State flows through all agents:\n"
            "   [dim]generated_number → doubled_number → final_response[/dim]\n\n"
            "4. Callbacks show exactly what's sent to/received from the LLM",
            border_style="green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
