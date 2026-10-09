"""Actions that must be confirmed by the user before they run.

The tool does NOT run the action. It stores it as "pending" and the agent asks the
user. If the user's next /chat message is "yes", the API runs it directly (no LLM
involved, so text on a web page can never approve an action).
"""
from __future__ import annotations

from typing import Callable

from app.context import ctx

EXECUTORS: dict[str, Callable[..., str]] = {}


def register(name: str):
    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        EXECUTORS[name] = fn
        return fn
    return deco


def request_confirmation(tool: str, args: dict, summary: str, continue_after: bool = False) -> str:
    """continue_after=True: after the user says yes, the agent continues the rest of the task."""
    ctx().pending_action = {"tool": tool, "args": args, "summary": summary, "continue": continue_after}
    return (f"CONFIRMATION REQUIRED: {summary}\n"
            "The action has NOT been run. Tell the user exactly what will happen and ask them to reply "
            "'yes' to confirm or 'no' to cancel. Do not call any more tools.")


def execute(action: dict) -> str:
    fn = EXECUTORS.get(action["tool"])
    if fn is None:
        return f"Error: no executor for {action['tool']}"
    return fn(**action["args"])
