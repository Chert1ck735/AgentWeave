"""Tool hook declarations, ToolContext injection and typed tool defaults."""
import pytest


def test_hooks_register_in_order_and_change_the_fingerprint():
    from agentweave import Agent, tool
    from agentweave.definitions import fingerprint
    from agentweave.hooks import Deny

    @tool
    def add(a: int, b: int) -> int:
        return a + b
    agent = Agent('assistant', tools=[add])
    before_hooks = fingerprint(agent)

    @agent.before_tool
    def first(call, ctx):
        return call

    @agent.before_tool
    async def second(call, ctx):
        return Deny('blocked')

    @agent.after_tool
    def shape(call, result, ctx):
        return result
    assert agent.before_tool_hooks == (first, second)
    assert agent.after_tool_hooks == (shape,)
    assert fingerprint(agent) != before_hooks
    assert Deny('x') == Deny('x') and Deny('x').reason == 'x'


def test_hook_decorator_rejects_non_callables():
    from agentweave import Agent
    with pytest.raises(ValueError):
        Agent('assistant').before_tool('not callable')


def test_tool_context_is_injected_not_part_of_the_schema():
    from agentweave import tool
    from agentweave.context import ToolContext

    @tool
    def stamp(value: str, ctx: ToolContext) -> str:
        return value + ctx.tool_call_id
    assert set(stamp.schema.input_schema['properties']) == {'value'}


def test_tool_defaults_are_validated_against_their_annotation():
    from agentweave import tool
    with pytest.raises(ValueError):
        @tool
        def bad(count: int = 'wrong') -> int:
            return count

    @tool
    def good(count: int = 3, label: str | None = None) -> int:
        return count
    assert good.schema.input_schema['properties']['count']['default'] == 3


def test_declarative_hook_references_resolve_from_project_modules(tmp_path):
    from agentweave.config import _resolve_agent
    from agentweave import Agent
    (tmp_path / 'guards.py').write_text(
        'from agentweave.hooks import Deny\n'
        'def deny_all(call, ctx):\n    return Deny("configured")\n')
    (tmp_path / 'agentweave.toml').write_text(
        '[agents.assistant]\nhooks = {before_tool = ["guards:deny_all"], after_tool = []}\n')
    resolved = _resolve_agent(Agent('assistant'), tmp_path)
    assert [hook.__name__ for hook in resolved.before_tool_hooks] == ['deny_all']
    assert resolved.after_tool_hooks == ()


def test_resolution_is_idempotent_for_toml_hooks_and_overrides(tmp_path):
    from agentweave import Agent
    from agentweave.config import _resolve_agent
    from agentweave.definitions import fingerprint
    (tmp_path / 'guards.py').write_text('def keep(call, ctx):\n    return call\n')
    (tmp_path / 'agentweave.toml').write_text(
        '[agents.assistant]\nconcurrency = 2\nhooks = {before_tool = ["guards:keep"]}\n')
    once = _resolve_agent(Agent('assistant'), tmp_path)
    twice = _resolve_agent(once, tmp_path)
    assert len(once.before_tool_hooks) == len(twice.before_tool_hooks) == 1
    assert fingerprint(twice) == fingerprint(once)
    overridden = _resolve_agent(once, tmp_path, {'concurrency': 4})
    assert overridden.concurrency == 4 and len(overridden.before_tool_hooks) == 1
    # A Worker resolves the definition it receives; an explicit override must survive that.
    again = _resolve_agent(overridden, tmp_path)
    assert again.concurrency == 4 and len(again.before_tool_hooks) == 1
