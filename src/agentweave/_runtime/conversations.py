"""Handoff: the responsibility for a task, and optionally a conversation, moves to another agent.

The task keeps its task_id and root; the store changes its owner, fencing and
the conversation's active agent in one transaction. The recipient gets the
portable part of the history (public text and closed tool call/result pairs)
plus the handoff message, never the sender's system prompt, other sessions or
a provider's opaque data. The root's deadline, expiry and model budget stay.
"""
from ..errors import HandoffRefused
from ..messages import Envelope, Message, validate_name
from ..models import ModelBudget

HANDOFF_TOOL = 'handoff_to'
MAX_HANDOFFS = 8


class HandoffSignal(BaseException):
    """Raised by ctx.handoff to end the handler; ordinary except Exception clauses do not stop it."""

    def __init__(self, agent: str, message: str | None, scope: str | None):
        super().__init__(agent)
        self.agent, self.message, self.scope = agent, message, scope


def handoff_tool(peers: list[str]):
    """The control tool a managed loop offers for handing off; it must be a turn's only call."""
    from ..definitions.tool import ToolSchema, schema_revision
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['agent', 'message'],
              'properties': {'agent': {'type': 'string', 'enum': sorted(peers)},
                             'message': {'type': 'string', 'description': 'what the next agent should know'}}}
    description = 'Hand this conversation to another agent, who answers from now on. Call it alone.'
    return ToolSchema(tool_id=HANDOFF_TOOL, description=description, input_schema=schema, output_schema={},
                      revision=schema_revision(schema, {}, description))


def portable(messages) -> list[Message]:
    """Public text and closed tool call/result pairs, without system prompts or adapter data."""
    kept = []
    items = list(messages)
    for index, message in enumerate(items):
        if message.role == 'system':
            continue
        if message.role == 'tool':
            if kept and kept[-1].role == 'assistant' and kept[-1].tool_calls:
                kept.append(Message(role='tool', tool_results=message.tool_results))
            continue
        calls = message.tool_calls if message.role == 'assistant' else []
        following = items[index + 1] if index + 1 < len(items) else None
        closed = bool(calls) and following is not None and following.role == 'tool' and len(following.tool_results) >= len(calls)
        kept.append(Message(role=message.role, content=message.content, tool_calls=calls if closed else []))
    return kept


def capable(directory, target: str, event: Envelope) -> bool:
    """target announces what this task needs: requests, or the same event type."""
    needed = 'request' if event.kind in ('request', 'consult') else event.topic or event.kind
    return any(entry.kind == 'agent' and needed in entry.capabilities
               for entry in directory.snapshot().get(target, ()))


def handoff_context(history, message: str | None, sender: str, budget: ModelBudget | None) -> dict:
    messages = portable(history)
    if message:
        messages.append(Message(role='assistant', content=message))
    return {'from': sender, 'budget': None if budget is None else budget.model_dump(mode='json'),
            'messages': [item.model_dump(mode='json') for item in messages]}


def transferred_messages(task) -> list[Message]:
    context = task.transfer_context if isinstance(task.transfer_context, dict) else {}
    return [Message.model_validate(item) for item in context.get('messages', [])]


def transferred_budget(task) -> ModelBudget | None:
    context = task.transfer_context if isinstance(task.transfer_context, dict) else {}
    try:
        return None if context.get('budget') is None else ModelBudget.model_validate(context['budget'])
    except ValueError:
        return None


async def hand_off(worker, lease, task, event: Envelope, target: str, context: dict, scope: str | None):
    """Check the recipient and move the task; HandoffRefused leaves the owner unchanged."""
    target = validate_name(target)
    if target == worker.definition.name:
        raise HandoffRefused('an agent cannot hand a task to itself')
    if not worker.policy.can_peer(worker.definition.name, target):
        raise HandoffRefused(f'{worker.definition.name} may not hand off to {target}')
    if not capable(worker.directory, target, event):
        raise HandoffRefused(f'{target} cannot take this task')
    conversation = await worker.store.get_conversation(task.session_id) if task.session_id else None
    owner = conversation is not None and conversation.active_agent == worker.definition.name
    if scope is None:
        # Only the conversation's owner passes the conversation on; a helper passes its task.
        scope = 'conversation' if owner else 'task'
    if scope == 'conversation' and not owner:
        raise HandoffRefused('only the conversation\'s active agent can hand off the conversation')
    return await worker.store.transfer(lease, target, context, scope, max_handoffs=MAX_HANDOFFS)


__all__ = ['HANDOFF_TOOL', 'HandoffSignal', 'handoff_tool', 'MAX_HANDOFFS', 'capable', 'hand_off', 'handoff_context', 'portable',
           'transferred_budget', 'transferred_messages']
