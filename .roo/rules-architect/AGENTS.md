# Architect Mode Rules (Non-Obvious Only)

- **Event-Driven**: Agents DO NOT return values. They yield `Event` streams. Architecture MUST handle this async stream.
- **State Isolation**: `session.state` is shared but mutable. Agents should write to specific keys (`output_key`) to avoid collisions.
- **Resumability**: Agents must explicitly signal completion (`end_of_agent=True`) to allow clean pause/resume of the session.
- **Hierarchy**: Parent agents must register sub-agents to manage their lifecycle and state context correctly.
- **Dependency Management**: `uv` handles dependencies. `agent-dev-handbook` is the package.
