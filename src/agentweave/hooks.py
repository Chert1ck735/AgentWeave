"""Programmatic tool checks that run before and after every tool effect.

A before hook returns the call to execute (possibly changed) or Deny(reason).
Changed arguments are validated again and journaled before the effect; the
call ID, tool, revision, caller, deadline and expiry are protected: a hook
cannot redirect a call to another tool, it can only deny it.
An after hook may reshape a successful result. It receives the call with the
arguments that actually ran; the originally requested arguments are in
ctx.requested_arguments. Its failure is journaled separately and never causes
the completed external effect to run again.
"""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from typing import Any
from .messages import JSONValue, ToolCall


@dataclass(frozen=True)
class Deny:
    """Returned by a before hook to refuse a tool call without any effect."""
    reason: str


BeforeTool = Callable[[ToolCall, Any], 'ToolCall | Deny | Awaitable[ToolCall | Deny]']
AfterTool = Callable[[ToolCall, JSONValue, Any], 'JSONValue | Awaitable[JSONValue]']


async def call_hook(hook: Callable, *arguments: Any) -> Any:
    """Run a sync or async hook inline; hooks are expected to be fast checks."""
    result = hook(*arguments)
    if inspect.isawaitable(result):
        result = await result
    return result


__all__ = ['AfterTool', 'BeforeTool', 'Deny', 'call_hook']
