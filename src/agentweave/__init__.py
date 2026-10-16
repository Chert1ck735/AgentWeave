"""Public declarations. Runtime/client exports arrive with their implementations."""
from .definitions import Agent, SubscriptionPolicy, Tools, Peers, tool
from .models import Model

__all__ = ['SubscriptionPolicy', 'Agent', 'Model', 'Tools', 'Peers', 'tool']


def __getattr__(name):
    if name in ('Client', 'AgentClient'):
        from .client import Client, AgentClient
        return {'Client':Client, 'AgentClient':AgentClient}[name]
    if name == 'Worker':
        from .worker import Worker
        return Worker
    if name == 'ToolWorker':
        from ._runtime.tool_service import ToolWorker
        return ToolWorker
    if name in ('Context', 'Request', 'ToolContext'):
        from .context import Context, Request, ToolContext
        return {'Context':Context, 'Request':Request, 'ToolContext':ToolContext}[name]
    raise AttributeError(name)

__all__ += ['Client', 'AgentClient', 'Worker', 'ToolWorker', 'Context', 'Request', 'ToolContext']
