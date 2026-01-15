"""
Callback Inspector Utility for Google ADK

This module provides utilities to inspect and debug what happens during
before_model_callback and after_model_callback in ADK agents.

Uses Rich library for beautiful terminal output.

Usage:
    from callback_inspector import (
        create_before_model_inspector,
        create_after_model_inspector,
        CallbackInspectorConfig,
    )

    agent = LlmAgent(
        name="my_agent",
        instruction="...",
        before_model_callback=create_before_model_inspector(),
        after_model_callback=create_after_model_inspector(),
    )
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Global console instance
console = Console()


@dataclass
class CallbackInspectorConfig:
    """Configuration for callback inspection."""

    verbose: bool = True
    log_state: bool = True
    log_tools: bool = True
    log_contents: bool = True
    log_usage: bool = True
    log_methods: bool = False
    log_dunders: bool = False
    custom_handler: Optional[Callable[["CallbackInspection"], None]] = None


@dataclass
class CallbackInspection:
    """Container for callback inspection results."""

    timestamp: str
    callback_type: str  # "before_model" or "after_model"
    agent_name: str
    invocation_id: str
    session_id: str
    user_id: str
    state_snapshot: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)


def _get_public_attributes(obj: Any) -> dict[str, Any]:
    """Extract public attributes from an object."""
    attrs = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
            if not callable(value):
                attrs[name] = _safe_repr(value)
        except Exception as e:
            attrs[name] = f"<error: {e}>"
    return attrs


def _get_methods(obj: Any) -> list[dict[str, str]]:
    """Extract method signatures from an object."""
    methods = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
            if callable(attr) and not isinstance(attr, type):
                sig = ""
                try:
                    sig = str(inspect.signature(attr))
                except (ValueError, TypeError):
                    sig = "(...)"
                methods.append(
                    {
                        "name": name,
                        "signature": sig,
                        "is_coroutine": inspect.iscoroutinefunction(attr),
                    }
                )
        except Exception:
            pass
    return methods


def _get_dunder_methods(obj: Any) -> list[str]:
    """Extract dunder (magic) methods from an object."""
    dunders = []
    for name in dir(obj):
        if name.startswith("__") and name.endswith("__"):
            if name not in (
                "__class__",
                "__doc__",
                "__module__",
                "__dict__",
                "__weakref__",
            ):
                try:
                    attr = getattr(type(obj), name, None)
                    if callable(attr):
                        dunders.append(name)
                except Exception:
                    pass
    return sorted(dunders)


def _safe_repr(value: Any, max_length: int = 100) -> str:
    """Safely convert a value to string representation."""
    try:
        if value is None:
            return "None"
        if isinstance(value, (str, int, float, bool)):
            result = repr(value)
            if len(result) > max_length:
                return result[: max_length - 3] + "..."
            return result
        if isinstance(value, dict):
            if len(value) == 0:
                return "{}"
            keys = list(value.keys())[:3]
            preview = ", ".join(f"{k!r}: ..." for k in keys)
            suffix = f" (+{len(value) - 3} more)" if len(value) > 3 else ""
            return f"{{{preview}{suffix}}}"
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "[]" if isinstance(value, list) else "()"
            bracket = "[]" if isinstance(value, list) else "()"
            return f"{bracket[0]}...{bracket[1]} ({len(value)} items)"
        result = repr(value)
        if len(result) > max_length:
            return result[: max_length - 3] + "..."
        return result
    except Exception as e:
        return f"<repr error: {e}>"


def inspect_callback_context(ctx: CallbackContext) -> dict[str, Any]:
    """Inspect a CallbackContext object and return detailed information."""
    return {
        "class": ctx.__class__.__name__,
        "module": ctx.__class__.__module__,
        "properties": {
            "agent_name": ctx.agent_name,
            "invocation_id": ctx.invocation_id,
            "user_id": ctx.user_id,
            "session_id": ctx.session.id if ctx.session else None,
            "app_name": ctx.session.app_name if ctx.session else None,
            "state_keys": list(ctx.state.keys()) if hasattr(ctx.state, "keys") else [],
            "run_config": _safe_repr(ctx.run_config),
            "user_content": _safe_repr(ctx.user_content),
        },
        "methods": _get_methods(ctx),
        "dunder_methods": _get_dunder_methods(ctx),
        "state_snapshot": dict(ctx.session.state) if ctx.session else {},
    }


def inspect_llm_request(request: LlmRequest) -> dict[str, Any]:
    """Inspect an LlmRequest object and return detailed information."""
    tools_info = {}
    if request.tools_dict:
        for name, tool in request.tools_dict.items():
            tools_info[name] = {
                "type": type(tool).__name__,
                "has_schema": hasattr(tool, "schema") and tool.schema is not None,
            }

    contents_summary = []
    if request.contents:
        for i, content in enumerate(request.contents):
            content_info = {"index": i, "role": getattr(content, "role", "unknown")}
            if hasattr(content, "parts") and content.parts:
                parts_info = []
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        text_preview = (
                            part.text[:80] + "..." if len(part.text) > 80 else part.text
                        )
                        parts_info.append({"type": "text", "preview": text_preview})
                    elif hasattr(part, "function_call") and part.function_call:
                        parts_info.append(
                            {
                                "type": "function_call",
                                "name": getattr(part.function_call, "name", "unknown"),
                            }
                        )
                    elif hasattr(part, "function_response") and part.function_response:
                        parts_info.append(
                            {
                                "type": "function_response",
                                "name": getattr(
                                    part.function_response, "name", "unknown"
                                ),
                            }
                        )
                    else:
                        parts_info.append({"type": type(part).__name__})
                content_info["parts"] = parts_info
            contents_summary.append(content_info)

    return {
        "class": request.__class__.__name__,
        "module": request.__class__.__module__,
        "properties": {
            "model": request.model,
            "num_contents": len(request.contents) if request.contents else 0,
            "num_tools": len(request.tools_dict) if request.tools_dict else 0,
            "has_config": request.config is not None,
            "cache_config": _safe_repr(request.cache_config),
            "previous_interaction_id": request.previous_interaction_id,
        },
        "methods": _get_methods(request),
        "dunder_methods": _get_dunder_methods(request),
        "tools": tools_info,
        "contents_summary": contents_summary,
    }


def inspect_llm_response(response: LlmResponse) -> dict[str, Any]:
    """Inspect an LlmResponse object and return detailed information."""
    content_summary = None
    if (
        response.content
        and hasattr(response.content, "parts")
        and response.content.parts
    ):
        parts_info = []
        for part in response.content.parts:
            if hasattr(part, "text") and part.text:
                text_preview = (
                    part.text[:150] + "..." if len(part.text) > 150 else part.text
                )
                parts_info.append({"type": "text", "preview": text_preview})
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                parts_info.append(
                    {
                        "type": "function_call",
                        "name": getattr(fc, "name", "unknown"),
                        "args": _safe_repr(getattr(fc, "args", {})),
                    }
                )
            else:
                parts_info.append({"type": type(part).__name__})
        content_summary = {
            "role": getattr(response.content, "role", "unknown"),
            "parts": parts_info,
        }

    return {
        "class": response.__class__.__name__,
        "module": response.__class__.__module__,
        "properties": {
            "model_version": response.model_version,
            "partial": response.partial,
            "turn_complete": response.turn_complete,
            "finish_reason": _safe_repr(response.finish_reason),
            "error_code": response.error_code,
            "error_message": response.error_message,
            "interrupted": response.interrupted,
            "has_grounding_metadata": response.grounding_metadata is not None,
            "has_usage_metadata": response.usage_metadata is not None,
            "avg_logprobs": response.avg_logprobs,
            "interaction_id": response.interaction_id,
        },
        "methods": _get_methods(response),
        "dunder_methods": _get_dunder_methods(response),
        "content_summary": content_summary,
        "usage": _extract_usage(response),
    }


def _extract_usage(response: LlmResponse) -> Optional[dict[str, Any]]:
    """Extract usage metadata from response if available."""
    if not response.usage_metadata:
        return None
    usage = response.usage_metadata
    return {
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
        "cached_content_token_count": getattr(
            usage, "cached_content_token_count", None
        ),
    }


def _render_before_inspection_rich(
    inspection: CallbackInspection, config: CallbackInspectorConfig
) -> None:
    """Render before_model inspection with Rich."""
    # Header
    header = Text()
    header.append("BEFORE MODEL CALLBACK", style="bold magenta")
    header.append(f" @ {inspection.timestamp}", style="dim")

    console.print()
    console.rule(header, style="magenta")

    # Context info table
    ctx_table = Table(show_header=False, box=None, padding=(0, 2))
    ctx_table.add_column("Key", style="cyan")
    ctx_table.add_column("Value", style="white")

    ctx_table.add_row("Agent", Text(inspection.agent_name, style="bold yellow"))
    ctx_table.add_row("Invocation", inspection.invocation_id[:16] + "...")
    ctx_table.add_row("Session", inspection.session_id)
    ctx_table.add_row("User", inspection.user_id)

    console.print(Panel(ctx_table, title="Context", border_style="blue"))

    # Session State
    if config.log_state and inspection.state_snapshot:
        state_tree = Tree("[bold green]Session State")
        for key, value in inspection.state_snapshot.items():
            state_tree.add(f"[cyan]{key}[/cyan]: [white]{_safe_repr(value)}[/white]")
        console.print(state_tree)

    # Request details
    req = inspection.details.get("request", {})
    req_table = Table(show_header=False, box=None, padding=(0, 2))
    req_table.add_column("Key", style="cyan")
    req_table.add_column("Value", style="white")

    req_table.add_row("Model", str(req.get("model", "N/A")))
    req_table.add_row("Contents", f"{req.get('num_contents', 0)} message(s)")
    req_table.add_row("Tools", f"{req.get('num_tools', 0)} available")

    console.print(Panel(req_table, title="LLM Request", border_style="green"))

    # Tools
    if config.log_tools and "tools" in inspection.details:
        tools = inspection.details["tools"]
        if tools:
            tools_tree = Tree("[bold blue]Available Tools")
            for tool_name, tool_info in tools.items():
                tools_tree.add(
                    f"[yellow]{tool_name}[/yellow] "
                    f"([dim]{tool_info.get('type', 'unknown')}[/dim])"
                )
            console.print(tools_tree)

    # Contents
    if config.log_contents and "contents" in inspection.details:
        contents = inspection.details["contents"]
        if contents:
            content_tree = Tree("[bold cyan]Message Contents")
            for content in contents:
                role = content.get("role", "unknown")
                role_style = "green" if role == "user" else "blue"
                content_node = content_tree.add(
                    f"[{role_style}]{role.upper()}[/{role_style}]"
                )
                for part in content.get("parts", []):
                    part_type = part.get("type", "unknown")
                    if part_type == "text":
                        preview = part.get("preview", "")[:60]
                        content_node.add(f"[dim]text:[/dim] {preview}...")
                    elif part_type == "function_call":
                        content_node.add(
                            f"[yellow]function_call:[/yellow] {part.get('name')}"
                        )
                    elif part_type == "function_response":
                        content_node.add(
                            f"[green]function_response:[/green] {part.get('name')}"
                        )
            console.print(content_tree)

    # Methods (if enabled)
    if config.log_methods:
        ctx_info = inspection.details.get("context_methods", [])

        if ctx_info:
            methods_table = Table(title="CallbackContext Methods", box=None)
            methods_table.add_column("Method", style="cyan")
            methods_table.add_column("Async", style="yellow")
            for m in ctx_info[:10]:
                async_marker = "yes" if m.get("is_coroutine") else ""
                methods_table.add_row(f"{m['name']}{m['signature']}", async_marker)
            console.print(methods_table)

    console.rule(style="magenta dim")


def _render_after_inspection_rich(
    inspection: CallbackInspection, config: CallbackInspectorConfig
) -> None:
    """Render after_model inspection with Rich."""
    # Header
    header = Text()
    header.append("AFTER MODEL CALLBACK", style="bold cyan")
    header.append(f" @ {inspection.timestamp}", style="dim")

    console.print()
    console.rule(header, style="cyan")

    # Context info
    ctx_table = Table(show_header=False, box=None, padding=(0, 2))
    ctx_table.add_column("Key", style="cyan")
    ctx_table.add_column("Value", style="white")

    ctx_table.add_row("Agent", Text(inspection.agent_name, style="bold yellow"))
    ctx_table.add_row("Invocation", inspection.invocation_id[:16] + "...")

    console.print(Panel(ctx_table, title="Context", border_style="blue"))

    # Response details
    resp = inspection.details.get("response", {})
    resp_table = Table(show_header=False, box=None, padding=(0, 2))
    resp_table.add_column("Key", style="cyan")
    resp_table.add_column("Value", style="white")

    resp_table.add_row("Model Version", str(resp.get("model_version", "N/A")))
    resp_table.add_row("Partial", str(resp.get("partial", "N/A")))
    resp_table.add_row("Turn Complete", str(resp.get("turn_complete", "N/A")))
    resp_table.add_row("Finish Reason", str(resp.get("finish_reason", "N/A")))

    if resp.get("has_error"):
        resp_table.add_row(
            Text("Error", style="bold red"), Text("Yes", style="bold red")
        )

    console.print(Panel(resp_table, title="LLM Response", border_style="green"))

    # Content summary
    content = inspection.details.get("content_summary")
    if content:
        content_tree = Tree(f"[bold green]Response Content ({content.get('role')})")
        for part in content.get("parts", []):
            part_type = part.get("type", "unknown")
            if part_type == "text":
                preview = part.get("preview", "")[:100]
                content_tree.add(f"[white]{preview}[/white]")
            elif part_type == "function_call":
                fn_name = part.get("name")
                fn_args = part.get("args", "{}")
                content_tree.add(f"[yellow]fn_call:[/yellow] {fn_name}({fn_args})")
        console.print(content_tree)

    # Usage metadata
    if config.log_usage:
        usage = inspection.details.get("usage")
        if usage:
            usage_table = Table(title="Token Usage", box=None)
            usage_table.add_column("Type", style="cyan")
            usage_table.add_column("Count", style="yellow", justify="right")

            if usage.get("prompt_token_count"):
                usage_table.add_row("Prompt", str(usage["prompt_token_count"]))
            if usage.get("candidates_token_count"):
                usage_table.add_row("Response", str(usage["candidates_token_count"]))
            if usage.get("total_token_count"):
                usage_table.add_row(
                    Text("Total", style="bold"), str(usage["total_token_count"])
                )
            if usage.get("cached_content_token_count"):
                usage_table.add_row("Cached", str(usage["cached_content_token_count"]))

            console.print(usage_table)

    # State changes
    if config.log_state and inspection.state_snapshot:
        state_tree = Tree("[bold green]Session State (after)")
        for key, value in inspection.state_snapshot.items():
            state_tree.add(f"[cyan]{key}[/cyan]: [white]{_safe_repr(value)}[/white]")
        console.print(state_tree)

    console.rule(style="cyan dim")


def create_before_model_inspector(
    config: Optional[CallbackInspectorConfig] = None,
) -> Callable[[CallbackContext, LlmRequest], None]:
    """
    Create a before_model_callback inspector with Rich output.

    Args:
        config: Configuration for the inspector. If None, uses defaults.

    Returns:
        A callback function compatible with before_model_callback
    """
    if config is None:
        config = CallbackInspectorConfig()

    def inspector(ctx: CallbackContext, request: LlmRequest) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        inspection = CallbackInspection(
            timestamp=timestamp,
            callback_type="before_model",
            agent_name=ctx.agent_name,
            invocation_id=ctx.invocation_id,
            session_id=ctx.session.id if ctx.session else "unknown",
            user_id=ctx.user_id,
            state_snapshot=dict(ctx.session.state)
            if ctx.session and config.log_state
            else {},
        )

        # Gather details
        inspection.details["context"] = {
            "agent_name": ctx.agent_name,
            "invocation_id": ctx.invocation_id,
            "user_id": ctx.user_id,
            "state_keys": list(ctx.state.keys()) if hasattr(ctx.state, "keys") else [],
        }

        inspection.details["request"] = {
            "model": request.model,
            "num_contents": len(request.contents) if request.contents else 0,
            "num_tools": len(request.tools_dict) if request.tools_dict else 0,
        }

        if config.log_tools and request.tools_dict:
            inspection.details["tools"] = {
                name: {"type": type(tool).__name__}
                for name, tool in request.tools_dict.items()
            }

        if config.log_contents and request.contents:
            inspection.details["contents"] = inspect_llm_request(request)[
                "contents_summary"
            ]

        if config.log_methods:
            inspection.details["context_methods"] = _get_methods(ctx)
            inspection.details["request_methods"] = _get_methods(request)

        if config.verbose:
            _render_before_inspection_rich(inspection, config)

        if config.custom_handler:
            config.custom_handler(inspection)

        return None

    return inspector


def create_after_model_inspector(
    config: Optional[CallbackInspectorConfig] = None,
) -> Callable[[CallbackContext, LlmResponse], None]:
    """
    Create an after_model_callback inspector with Rich output.

    Args:
        config: Configuration for the inspector. If None, uses defaults.

    Returns:
        A callback function compatible with after_model_callback
    """
    if config is None:
        config = CallbackInspectorConfig()

    def inspector(ctx: CallbackContext, response: LlmResponse) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        inspection = CallbackInspection(
            timestamp=timestamp,
            callback_type="after_model",
            agent_name=ctx.agent_name,
            invocation_id=ctx.invocation_id,
            session_id=ctx.session.id if ctx.session else "unknown",
            user_id=ctx.user_id,
            state_snapshot=dict(ctx.session.state)
            if ctx.session and config.log_state
            else {},
        )

        # Gather details
        inspection.details["context"] = {
            "agent_name": ctx.agent_name,
            "invocation_id": ctx.invocation_id,
            "state_keys": list(ctx.state.keys()) if hasattr(ctx.state, "keys") else [],
        }

        inspection.details["response"] = {
            "model_version": response.model_version,
            "partial": response.partial,
            "turn_complete": response.turn_complete,
            "finish_reason": str(response.finish_reason)
            if response.finish_reason
            else None,
            "has_error": response.error_code is not None,
        }

        response_info = inspect_llm_response(response)
        inspection.details["content_summary"] = response_info.get("content_summary")

        if config.log_usage:
            inspection.details["usage"] = _extract_usage(response)

        if config.verbose:
            _render_after_inspection_rich(inspection, config)

        if config.custom_handler:
            config.custom_handler(inspection)

        return None

    return inspector


def print_workflow_header(title: str, subtitle: str = "") -> None:
    """Print a styled workflow header."""
    console.print()
    console.print(
        Panel(
            f"[bold white]{title}[/bold white]\n[dim]{subtitle}[/dim]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )


def print_agent_transition(from_agent: str, to_agent: str) -> None:
    """Print agent transition marker."""
    console.print()
    console.print(
        f"  [dim]──────[/dim] [yellow]{from_agent}[/yellow] "
        f"[dim]→[/dim] [green]{to_agent}[/green] [dim]──────[/dim]"
    )


def print_event(author: str, content: str, event_type: str = "text") -> None:
    """Print a styled event."""
    style_map = {
        "text": "white",
        "function_call": "yellow",
        "function_response": "green",
        "error": "red",
    }
    style = style_map.get(event_type, "white")

    author_style = "cyan" if "llm" in author.lower() else "magenta"
    console.print(
        f"  [{author_style}]{author}[/{author_style}]: [{style}]{content}[/{style}]"
    )


def print_state_table(state: dict[str, Any], title: str = "Session State") -> None:
    """Print session state as a formatted table."""
    table = Table(title=title, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    for key, value in state.items():
        table.add_row(key, _safe_repr(value, max_length=60))

    console.print(table)


def print_summary(stats: dict[str, Any]) -> None:
    """Print workflow summary statistics."""
    console.print()
    console.print(Panel("[bold]Workflow Summary[/bold]", border_style="green"))

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow", justify="right")

    for key, value in stats.items():
        table.add_row(key, str(value))

    console.print(table)


# Export public API
__all__ = [
    "CallbackInspection",
    "CallbackInspectorConfig",
    "inspect_callback_context",
    "inspect_llm_request",
    "inspect_llm_response",
    "create_before_model_inspector",
    "create_after_model_inspector",
    "print_workflow_header",
    "print_agent_transition",
    "print_event",
    "print_state_table",
    "print_summary",
    "console",
]
