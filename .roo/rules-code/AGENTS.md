# Code Mode Rules (Non-Obvious Only)

- **Sub-Agent Registration**: `super().__init__(..., sub_agents=[list_of_agents])` is MANDATORY. Failure results in broken hierarchy/state.
- **Yielding Events**: EVERY agent method must yield `Event` objects. Do NOT return values.
- **Streaming Pattern**: Use `async with Aclosing(...) as agen: async for event in agen: yield event`. Never collect into list first.
- **State Keys**: Ensure `output_key` in LLM agents matches placeholders in downstream prompts (e.g., `{story}` needs `output_key="story"`).
- **Runnable Scripts**: "Tests" in `examples/` are scripts. Run with `python filename.py`. Do NOT try to fix them to work with pytest.
