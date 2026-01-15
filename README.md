# Agent Development Handbook

This repository serves as a comprehensive guide to building AI agents using modern frameworks and best practices. It provides educational implementations, examples, and patterns for developing agentic systems, with a primary focus on LangGraph and Google's Agent Development Kit (ADK).

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

### Prerequisites

Install uv if you haven't already:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/whanyu1212/agent-dev-handbook.git
cd agent-dev-handbook

# Install dependencies (creates .venv automatically)
uv sync

# Install with dev dependencies for testing and code quality tools
uv sync --all-extras
```

### Running Examples

Activate the virtual environment and explore the examples:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Run examples (once implemented)
python -m agent_dev_handbook.langgraph.examples.<example_name>
python -m agent_dev_handbook.adk.examples.<example_name>
```

### Development

```bash
# Run linter with auto-fix
ruff check --fix .

# Format code
ruff format .

# Run tests
pytest

# Run pre-commit hooks
pre-commit run --all-files
```

## What's Inside

This handbook covers practical agent development through:

- **LangGraph**: Building stateful, graph-based agents with cyclic workflows
- **Google ADK**: Leveraging Google's Agent Development Kit for scalable agent architectures
- Code examples and reference implementations
- Best practices and design patterns for agentic systems

## Focus Areas

- Agent architecture and design patterns
- State management in multi-step workflows
- Tool integration and function calling
- Memory and context management
- Evaluation and testing strategies for agents

## AI-Assisted Development

This repository includes guidance files for AI coding assistants to help them understand the codebase and follow project conventions:

- **`CLAUDE.md`** - Instructions for [Claude Code](https://claude.ai/code) with project context, development commands, architecture overview, key patterns, and common pitfalls
- **`AGENTS.md`** - Concise guidance for any AI agent with essential commands, code patterns, and gotchas

These files help AI tools provide more accurate and context-aware assistance when working with this codebase. If you use AI-assisted development tools (Claude Code, Cursor, GitHub Copilot, etc.), these files will help them better understand the project structure and ADK-specific patterns like event-driven architecture and state management.
