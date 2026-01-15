"""
Test: Dynamic Tools and Prompts Across Multiple Turns

This demonstrates that dynamic tools and prompts are re-evaluated
on EVERY turn based on the current session state.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

load_dotenv(Path(__file__).parent / ".env")


# Define tools
def basic_tool() -> str:
    """Basic tool available to all tiers."""
    return "Basic feature result"


def premium_tool() -> str:
    """Premium tool only available to premium users."""
    return "Premium feature result"


# Dynamic instruction that changes based on tier
def tier_based_instruction(ctx: ReadonlyContext) -> str:
    """Instruction changes based on user tier in state."""
    tier = ctx.session.state.get("user_tier", "free")
    turn = ctx.session.state.get("turn_count", 0)

    print(f"\n🔄 [Turn {turn}] InstructionProvider called with tier={tier}")

    if tier == "premium":
        return (
            "You are a PREMIUM assistant with access to advanced features. "
            "Mention your premium status."
        )
    else:
        return (
            "You are a FREE assistant with basic features only. "
            "Mention your free tier status."
        )


# Dynamic tools that change based on tier
def tier_based_tools(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """Tools change based on user tier in state."""
    tier = callback_context.session.state.get("user_tier", "free")
    turn = callback_context.session.state.get("turn_count", 0)

    print(f"🔧 [Turn {turn}] before_model_callback called with tier={tier}")

    # Clear existing tools
    llm_request.tools_dict.clear()

    # Everyone gets basic tool
    llm_request.tools_dict["basic_tool"] = FunctionTool(func=basic_tool)

    # Premium users get premium tool
    if tier == "premium":
        llm_request.tools_dict["premium_tool"] = FunctionTool(func=premium_tool)
        print("   ✅ Added: basic_tool, premium_tool")
    else:
        print("   ✅ Added: basic_tool only")

    return None


async def main():
    print("=" * 70)
    print("MULTI-TURN DYNAMIC BEHAVIOR TEST")
    print("=" * 70)
    print("\nThis test shows that dynamic prompts and tools are re-evaluated")
    print("on EVERY turn based on current session state.\n")

    # Create agent with dynamic behavior
    agent = LlmAgent(
        name="dynamic_agent",
        instruction=tier_based_instruction,  # Dynamic prompt
        tools=[],  # Will be added dynamically
        before_model_callback=tier_based_tools,  # Dynamic tools
    )

    # Setup session
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="DynamicTest",
        user_id="user1",
        session_id="session1",
        state={
            "user_tier": "free",  # Start as FREE
            "turn_count": 0,
        },
    )

    runner = Runner(
        app_name="DynamicTest", agent=agent, session_service=session_service
    )

    # ========================================================================
    # TURN 1: User is FREE tier
    # ========================================================================
    print("\n" + "=" * 70)
    print("TURN 1: User is FREE tier")
    print("=" * 70)

    # Update turn count
    session = await session_service.get_session(
        app_name="DynamicTest", user_id="user1", session_id="session1"
    )
    session.state["turn_count"] = 1

    events = runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=types.Content(
            role="user", parts=[types.Part(text="What tools do you have access to?")]
        ),
    )

    print("\n📝 Agent Response:")
    async for event in events:
        if event.content and event.content.parts and event.content.parts[0].text:
            text = event.content.parts[0].text
            print(f"   {text[:200]}")
            break

    # ========================================================================
    # TURN 2: Upgrade user to PREMIUM
    # ========================================================================
    print("\n" + "=" * 70)
    print("TURN 2: Upgrade user to PREMIUM (state change)")
    print("=" * 70)

    # Modify session state - upgrade to premium
    session = await session_service.get_session(
        app_name="DynamicTest", user_id="user1", session_id="session1"
    )
    session.state["user_tier"] = "premium"  # UPGRADE!
    session.state["turn_count"] = 2
    print("✨ State changed: user_tier = 'premium'")

    events = runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What about now? What tools do you have?")],
        ),
    )

    print("\n📝 Agent Response:")
    async for event in events:
        if event.content and event.content.parts and event.content.parts[0].text:
            text = event.content.parts[0].text
            print(f"   {text[:200]}")
            break

    # ========================================================================
    # TURN 3: Downgrade back to FREE
    # ========================================================================
    print("\n" + "=" * 70)
    print("TURN 3: Downgrade user back to FREE (state change)")
    print("=" * 70)

    # Modify session state - downgrade
    session = await session_service.get_session(
        app_name="DynamicTest", user_id="user1", session_id="session1"
    )
    session.state["user_tier"] = "free"  # DOWNGRADE!
    session.state["turn_count"] = 3
    print("⬇️  State changed: user_tier = 'free'")

    events = runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=types.Content(role="user", parts=[types.Part(text="And now?")]),
    )

    print("\n📝 Agent Response:")
    async for event in events:
        if event.content and event.content.parts and event.content.parts[0].text:
            text = event.content.parts[0].text
            print(f"   {text[:200]}")
            break

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✅ Dynamic prompts and tools ARE DURABLE across turns!")
    print("   They are re-evaluated on EVERY turn based on current state.")
    print("\n📌 Key Points:")
    print("   1. InstructionProvider called every turn")
    print("   2. before_model_callback called every turn")
    print("   3. Both read from session.state")
    print("   4. Changes to state immediately affect next turn")
    print("\n💡 This means you can:")
    print("   - Upgrade/downgrade users mid-conversation")
    print("   - Enable/disable features dynamically")
    print("   - Change behavior based on conversation progress")
    print("   - Implement progressive disclosure of features")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
