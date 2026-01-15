"""
Demo: Using the Callback Inspector Utility

This script demonstrates how to use the callback_inspector module
to inspect what happens during before_model_callback and after_model_callback.
"""

import asyncio
from pathlib import Path

from callback_inspector import (
    create_after_model_inspector,
    create_before_model_inspector,
    inspect_callback_context,
    inspect_llm_request,
    inspect_llm_response,
    print_object_inspection,
)
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(Path(__file__).parent / ".env")


# Example 1: Simple inspection with verbose output
def demo_simple_inspection():
    """Create an agent with verbose callback inspectors."""
    print("\n" + "=" * 70)
    print("DEMO 1: Simple Callback Inspection")
    print("=" * 70)

    agent = LlmAgent(
        name="inspected_agent",
        instruction="You are a helpful assistant. Keep responses brief.",
        before_model_callback=create_before_model_inspector(
            verbose=True,
            log_state=True,
            log_tools=True,
            log_contents=True,
        ),
        after_model_callback=create_after_model_inspector(
            verbose=True,
            log_state=True,
            log_content=True,
            log_usage=True,
        ),
    )

    return agent


# Example 2: Custom handler to collect inspection data
collected_inspections: list[dict] = []


def custom_inspection_handler(inspection):
    """Custom handler that collects inspection data."""
    collected_inspections.append(
        {
            "timestamp": inspection.timestamp,
            "type": inspection.callback_type,
            "agent": inspection.agent_name,
            "state": inspection.state_snapshot,
            "details": inspection.details,
        }
    )
    print(f"  📌 Custom handler collected {inspection.callback_type} inspection")


def demo_custom_handler():
    """Create an agent with custom inspection handler."""
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Inspection Handler")
    print("=" * 70)

    agent = LlmAgent(
        name="custom_handler_agent",
        instruction="You are a helpful assistant.",
        before_model_callback=create_before_model_inspector(
            verbose=False,  # Quiet mode
            custom_handler=custom_inspection_handler,
        ),
        after_model_callback=create_after_model_inspector(
            verbose=False,  # Quiet mode
            custom_handler=custom_inspection_handler,
        ),
    )

    return agent


# Example 3: Manual inspection within a callback
def demo_manual_inspection():
    """Show how to manually inspect objects within a callback."""
    print("\n" + "=" * 70)
    print("DEMO 3: Manual Object Inspection")
    print("=" * 70)

    def detailed_before_callback(ctx, request):
        """Manually inspect all available data."""
        print("\n🔬 Manual inspection in before_model_callback:")

        # Inspect the context
        ctx_info = inspect_callback_context(ctx)
        print(f"\n  CallbackContext class: {ctx_info['class']}")
        print(f"  Properties: {list(ctx_info['properties'].keys())}")
        print(f"  Available methods: {[m['name'] for m in ctx_info['methods']]}")
        print(f"  Dunder methods: {ctx_info['dunder_methods'][:10]}...")

        # Inspect the request
        req_info = inspect_llm_request(request)
        print(f"\n  LlmRequest class: {req_info['class']}")
        print(f"  Properties: {list(req_info['properties'].keys())}")
        print(f"  Available methods: {[m['name'] for m in req_info['methods']]}")
        tools_list = list(req_info["tools"].keys()) if req_info["tools"] else "None"
        print(f"  Tools: {tools_list}")

        return None

    def detailed_after_callback(ctx, response):
        """Manually inspect response data."""
        print("\n🔬 Manual inspection in after_model_callback:")

        # Inspect the response
        resp_info = inspect_llm_response(response)
        print(f"\n  LlmResponse class: {resp_info['class']}")
        print(f"  Properties: {list(resp_info['properties'].keys())}")
        print(f"  Available methods: {[m['name'] for m in resp_info['methods']]}")
        print(f"  Usage: {resp_info['usage']}")

        # Use print_object_inspection for detailed view
        print("\n  Full object inspection:")
        print_object_inspection(
            response, "LlmResponse", show_methods=True, show_dunders=False
        )

        return None

    agent = LlmAgent(
        name="manual_inspection_agent",
        instruction="You are a helpful assistant.",
        before_model_callback=detailed_before_callback,
        after_model_callback=detailed_after_callback,
    )

    return agent


async def run_agent(agent: LlmAgent, message: str):
    """Run an agent with a single message."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="InspectorDemo",
        user_id="user1",
        session_id="session1",
        state={"demo_key": "demo_value", "counter": 0},
    )

    runner = Runner(
        app_name="InspectorDemo",
        agent=agent,
        session_service=session_service,
    )

    print(f"\n📤 Sending message: '{message}'")

    events = runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=message)],
        ),
    )

    print("\n📥 Agent Response:")
    async for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(f"   {part.text[:300]}")
                    return


async def main():
    print("=" * 70)
    print("CALLBACK INSPECTOR DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how to use the callback_inspector module to")
    print("understand what data is available during ADK callbacks.\n")

    # Demo 1: Simple verbose inspection
    agent1 = demo_simple_inspection()
    await run_agent(agent1, "Hello! What can you do?")

    # Demo 2: Custom handler
    agent2 = demo_custom_handler()
    await run_agent(agent2, "Tell me a joke.")

    print("\n📊 Collected inspections from custom handler:")
    for i, insp in enumerate(collected_inspections):
        print(f"   [{i}] {insp['type']} at {insp['timestamp']}")

    # Demo 3: Manual inspection
    agent3 = demo_manual_inspection()
    await run_agent(agent3, "What is 2+2?")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Available Inspection Functions")
    print("=" * 70)
    print("""
    1. create_before_model_inspector() - Factory for before_model_callback
       - verbose: Print to console
       - log_state: Include session state
       - log_tools: Include tool information
       - log_contents: Include message contents
       - custom_handler: Your own callback function

    2. create_after_model_inspector() - Factory for after_model_callback
       - verbose: Print to console
       - log_state: Include session state
       - log_content: Include response content
       - log_usage: Include token usage
       - custom_handler: Your own callback function

    3. inspect_callback_context(ctx) - Get dict of context info
       - properties, methods, dunder_methods, state_snapshot

    4. inspect_llm_request(request) - Get dict of request info
       - properties, methods, tools, contents_summary

    5. inspect_llm_response(response) - Get dict of response info
       - properties, methods, content_summary, usage

    6. print_object_inspection(obj) - Print detailed object inspection

    7. get_full_inspection(ctx, request, response) - Get everything as dict
    """)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
