"""Frozen programmatic scopes. Call on every dispatch, including model choices."""
from collections.abc import Iterable, Mapping
from .definitions import Agent
from .definitions.selections import ToolSelection
from .errors import DefinitionError
from .messages import validate_name


class Policy:
    def __init__(self, agents: Iterable[Agent], *,
                 tool_callers: Mapping[str, Iterable[str] | None] | None = None):
        self._peers, self._tools, self._local_callers = {}, {}, {}
        self._tool_callers = {validate_name(tool_id): None if callers is None else
            frozenset(validate_name(caller) for caller in callers)
            for tool_id, callers in (tool_callers or {}).items()}
        for agent in agents:
            if agent.name in self._peers:
                raise DefinitionError('duplicate policy caller')
            peers = agent.peers
            self._peers[agent.name] = None if peers is not None and peers.mode == 'discover' else frozenset(peers.names if peers else ())
            tools = agent.tools
            if isinstance(tools, ToolSelection):
                self._tools[agent.name] = None if tools.allow is None else tuple(tools.allow)
            else:
                self._tools[agent.name] = tuple(tool.name for tool in (tools or ()))
                for tool in tools or ():
                    self._local_callers[(agent.name, tool.name)] = None if tool.allowed_callers is None else frozenset(tool.allowed_callers)

    @staticmethod
    def _valid(caller, target):
        try:
            validate_name(caller)
            validate_name(target)
            return True
        except ValueError:
            return False

    def can_peer(self, caller: str, target: str) -> bool:
        if not self._valid(caller, target) or caller not in self._peers:
            return False
        scope = self._peers[caller]
        return scope is None or target in scope

    def can_tool(self, caller: str, tool_id: str) -> bool:
        if not self._valid(caller, tool_id) or caller not in self._tools:
            return False
        scope = self._tools[caller]
        granted = scope is None or any(tool_id == pattern or
            (pattern.endswith('.*') and tool_id.startswith(pattern[:-1])) for pattern in scope)
        callers = self._tool_callers.get(tool_id)
        local_callers = self._local_callers.get((caller, tool_id))
        return granted and (callers is None or caller in callers) and (local_callers is None or caller in local_callers)
