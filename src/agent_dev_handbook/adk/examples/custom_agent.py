"""
Custom Agent Example - Conditional Story Generator

This example demonstrates how to build a custom agent that:
1. Generates a story based on user input
2. Analyzes the story's sentiment
3. Conditionally regenerates if sentiment is negative
4. Uses session state to pass data between stages
"""

from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.utils.context_utils import Aclosing


class ConditionalStoryAgent(BaseAgent):
    """
    Custom agent that orchestrates story generation with conditional logic.

    Architecture:
    1. story_generator: Creates initial story
    2. sentiment_analyzer: Analyzes story sentiment
    3. Conditional: If negative sentiment, regenerate story

    State keys used:
    - topic: Input topic for story (set by user)
    - current_story: The generated story
    - sentiment: Result of sentiment analysis ("positive" or "negative")
    """

    # Type hints for sub-agents
    story_generator: LlmAgent
    sentiment_analyzer: LlmAgent

    def __init__(
        self,
        name: str,
        story_generator: LlmAgent,
        sentiment_analyzer: LlmAgent,
    ):
        """
        Initialize the custom agent.

        Args:
            name: Agent name (must be unique identifier)
            story_generator: LlmAgent that creates stories
            sentiment_analyzer: LlmAgent that analyzes sentiment
        """
        # Store sub-agents as instance attributes
        self.story_generator = story_generator
        self.sentiment_analyzer = sentiment_analyzer

        # Register sub-agents with parent BaseAgent
        # This is REQUIRED for proper agent hierarchy
        super().__init__(
            name=name,
            sub_agents=[story_generator, sentiment_analyzer],
            description="Generates stories with sentiment-based regeneration",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Core orchestration logic for text-based conversation.

        This method demonstrates:
        - Sequential sub-agent execution
        - Reading/writing session state
        - Conditional logic based on agent outputs
        - Proper event streaming with Aclosing
        - Agent completion signaling

        Args:
            ctx: Invocation context containing session state and event history

        Yields:
            Event objects from sub-agents and state updates
        """

        # Stage 1: Generate initial story
        # The story_generator will save output to state["current_story"]
        async with Aclosing(self.story_generator.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # Stage 2: Analyze sentiment of generated story
        # The sentiment_analyzer will save output to state["sentiment"]
        async with Aclosing(self.sentiment_analyzer.run_async(ctx)) as agen:
            async for event in agen:
                yield event

        # Stage 3: Conditional regeneration based on sentiment
        sentiment = ctx.session.state.get("sentiment", "").lower()

        if "negative" in sentiment:
            # Sentiment is negative, regenerate the story
            # You could add a different prompt or instruction here
            async with Aclosing(self.story_generator.run_async(ctx)) as agen:
                async for event in agen:
                    yield event

        # Mark this agent as complete (important for resumability)
        if ctx.is_resumable:
            ctx.set_agent_state(self.name, end_of_agent=True)
            yield self._create_agent_state_event(ctx)


# Example usage and sub-agent definitions
if __name__ == "__main__":
    # Define the story generator agent
    story_generator = LlmAgent(
        name="story_generator",
        instruction=(
            "Generate a creative short story (3-4 sentences) about: {topic}\n"
            "Make it engaging and imaginative."
        ),
        output_key="current_story",  # Save output to state["current_story"]
    )

    # Define the sentiment analyzer agent
    sentiment_analyzer = LlmAgent(
        name="sentiment_analyzer",
        instruction=(
            "Analyze the sentiment of this story: {current_story}\n\n"
            "Respond with only one word: 'positive' or 'negative'"
        ),
        output_key="sentiment",  # Save output to state["sentiment"]
    )

    # Create the custom orchestrator agent
    story_agent = ConditionalStoryAgent(
        name="conditional_story_agent",
        story_generator=story_generator,
        sentiment_analyzer=sentiment_analyzer,
    )

    # Example of how to run this agent:
    """
    from google.adk.runners.runner import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    # Setup
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="StoryApp",
        user_id="user123",
        session_id="session456",
        state={"topic": "a lonely robot"}
    )

    # Run
    runner = Runner(agent=story_agent, session_service=session_service)
    events = runner.run_async(user_id="user123", session_id="session456")

    # Process results
    async for event in events:
        if event.is_final_response():
            print(event.content.parts[0].text)

    # Check final state
    final_session = await session_service.get_session(
        app_name="StoryApp", user_id="user123", session_id="session456"
    )
    print(f"Story: {final_session.state['current_story']}")
    print(f"Sentiment: {final_session.state['sentiment']}")
    """
