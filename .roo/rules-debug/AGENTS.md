# Debug Mode Rules (Non-Obvious Only)

- **"Tests" are Scripts**: Do NOT run `pytest`. Run example scripts directly: `python src/agent_dev_handbook/adk/examples/test_dynamic_multiturn.py`.
- **Missing Test Dir**: The `test/` directory referenced in `pyproject.toml` DOES NOT EXIST.
- **Event Tracing**: `session.events` contains the immutable history. Inspect it to debug conversation flow.
- **State Inspection**: `ctx.session.state` holds the current snapshot. Check it if agents "forget" context.
- **Silent Failures**: If streaming stops, check if `Aclosing` was used. Missing it causes silent generator cleanup issues.
