"""Permissions are declaration scopes, independent of catalog visibility."""
from agentweave import Agent, Peers, Tools


def policy(*agents, **kwargs):
    from agentweave._policy import Policy
    return Policy(agents, **kwargs)


def test_peer_discovery_does_not_grant_tools():
    scopes = policy(Agent('assistant', peers=Peers.discover()))
    assert scopes.can_peer('assistant', 'analyst') is True
    assert scopes.can_tool('assistant', 'calculator.add') is False


def test_empty_allow_closes_everything_and_unknown_caller_is_denied():
    scopes = policy(Agent('assistant', peers=Peers.allow([]), tools=Tools.discover(allow=[])))
    assert not scopes.can_peer('assistant', 'analyst')
    assert not scopes.can_tool('assistant', 'calculator.add')
    assert not scopes.can_peer('unknown', 'analyst')
    assert not scopes.can_tool('unknown', 'calculator.add')


def test_peer_allow_is_exact_and_tools_have_independent_namespace_allow():
    scopes = policy(Agent('assistant', peers=Peers.allow(['analyst']),
        tools=Tools.discover(allow=['calculator.*', 'lookup'])))
    assert scopes.can_peer('assistant', 'analyst')
    assert not scopes.can_peer('assistant', 'analyst.other')
    assert scopes.can_tool('assistant', 'calculator.add')
    assert scopes.can_tool('assistant', 'calculator.nested.add')
    assert scopes.can_tool('assistant', 'lookup')
    assert not scopes.can_tool('assistant', 'calculator')
    assert not scopes.can_tool('assistant', 'calculator2.add')


def test_tool_discovery_and_executor_allow_must_both_permit_caller():
    allowed = {'calculator.add': ['assistant'], 'calculator.secret': []}
    scopes = policy(Agent('assistant', tools=Tools.discover()),
        Agent('analyst', tools=Tools.discover()), tool_callers=allowed)
    assert scopes.can_tool('assistant', 'calculator.add')
    assert not scopes.can_tool('analyst', 'calculator.add')
    assert not scopes.can_tool('assistant', 'calculator.secret')
    allowed['calculator.add'].clear()
    assert scopes.can_tool('assistant', 'calculator.add')
    assert not scopes.can_peer('assistant', 'analyst')


def test_local_tools_do_not_grant_other_tools_and_enforce_allowed_callers():
    from agentweave import tool
    @tool
    def add(x: int, y: int) -> int:
        return x + y
    restricted = add.model_copy(update={'allowed_callers': ('assistant',)})
    scopes = policy(Agent('assistant', tools=[restricted]), Agent('other', tools=[restricted]))
    assert scopes.can_tool('assistant', 'add')
    assert not scopes.can_tool('assistant', 'calculator.add')
    assert not scopes.can_tool('other', 'add')


def test_invalid_target_cannot_match_open_scope():
    scopes = policy(Agent('assistant', peers=Peers.discover(), tools=Tools.discover()))
    assert not scopes.can_peer('assistant', '../other')
    assert not scopes.can_tool('assistant', '')
