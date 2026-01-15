# Code Mode Rules (Non-Obvious Only)

## Linting (MUST follow to avoid errors)
- **Line length**: 88 chars max - split long strings or extract to variables
- **Unused imports** (F401): Remove them immediately
- **Unused variables** (F841): Remove or prefix with `_`
- **Import order** (I001): stdlib → third-party → local, alphabetized within groups
- **Ruff config**: `select = ["E", "F", "I"]` (pycodestyle + pyflakes + isort)

## ADK Patterns
- **Sub-Agent Registration**: `super().__init__(..., sub_agents=[list_of_agents])` is MANDATORY. Failure results in broken hierarchy/state.
- **Yielding Events**: EVERY agent method must yield `Event` objects. Do NOT return values.
- **Streaming Pattern**: Use `async with Aclosing(...) as agen: async for event in agen: yield event`. Never collect into list first.
- **State Keys**: Ensure `output_key` in LLM agents matches placeholders in downstream prompts (e.g., `{story}` needs `output_key="story"`).

## Project Notes
- **Runnable Scripts**: "Tests" in `examples/` are scripts. Run with `python filename.py`. Do NOT try to fix them to work with pytest.
- **Mypy**: Uses `--ignore-missing-imports --no-strict-optional` (lenient for educational examples).
