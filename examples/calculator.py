"""A calculator published as an independent tool service.

    agentweave dev run examples.calculator:add --namespace calculator --detach

publishes calculator.add in the live catalog. An agent with
tools=Tools.discover() offers it to its model on its next turn without a
restart and without importing this module; the call travels through the
broker and the journal, and this process executes it.
"""
from agentweave import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b
