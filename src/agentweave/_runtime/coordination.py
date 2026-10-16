"""Consulting peers: one agent's task asks another agent and continues with the answer.

A consult is a child request task: it carries only the explicit input, a new
session, the parent's deadline and expiry, the root's model budget and the
chain of waiting ancestors (to refuse cycles). The parent waits durably: a
managed loop gives up its lease and slot until every child ended, a Python
handler keeps its lease but frees its slot. The root's model counter and
budget are shared because children belong to the same root task.
"""
import asyncio
import logging
from uuid import UUID, uuid5
from .._records import TaskRecord
from ..definitions.tool import ToolSchema, schema_revision
from ..errors import ConsultRefused, CycleDetected, ReceiptNotExpected, StaleDelivery
from ..messages import Envelope, JSONValue, TaskResult, public_error, validate_name
from ..models import ModelBudget

_logger = logging.getLogger('agentweave.runtime')
CONSULT_TOOL = 'consult_agent'
MAX_CONSULT_DEPTH = 8
CHILD_POLL_SECONDS = 0.05
_TERMINAL = frozenset(('succeeded', 'failed', 'cancelled', 'expired', 'outcome_unknown'))


def request_input(event: Envelope, task_input: JSONValue) -> JSONValue:
    """What the receiving handler or model sees: a consult's explicit input, a request's payload."""
    if event.kind == 'consult' and isinstance(task_input, dict):
        return task_input.get('input')
    return task_input


def ancestry(event: Envelope) -> list[str]:
    """The agents waiting above this task, root first."""
    if event.kind == 'consult' and isinstance(event.payload, dict):
        return [name for name in event.payload.get('ancestry', []) if isinstance(name, str)]
    return []


def inherited_budget(event: Envelope) -> ModelBudget | None:
    """A consult inherits the root's model budget instead of starting a new one."""
    if event.kind == 'consult' and isinstance(event.payload, dict) and event.payload.get('budget') is not None:
        try:
            return ModelBudget.model_validate(event.payload['budget'])
        except ValueError:
            _logger.warning('ignoring a malformed inherited budget', extra={'task_id': event.task_id})
    return None


def consult_tool(peers: list[str]) -> ToolSchema:
    """The tool a managed loop offers for consulting; only allowed live peers can be named."""
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['agent', 'input'],
              'properties': {'agent': {'type': 'string', 'enum': sorted(peers)},
                             'input': {'description': 'the request for that agent'}}}
    description = 'Ask another agent of this network and continue with its answer.'
    return ToolSchema(tool_id=CONSULT_TOOL, description=description, input_schema=schema, output_schema={},
                      revision=schema_revision(schema, {}, description))


def live_peers(directory, policy, sender: str) -> list[str]:
    """Agents that answer requests now and that sender may consult."""
    return sorted(name for name, instances in directory.snapshot().items()
                  if name != sender and instances[0].kind == 'agent' and 'request' in instances[0].capabilities
                  and policy.can_peer(sender, name))


def refusal(directory, policy, sender: str, target: str, chain: list[str]) -> tuple[str, str] | None:
    """Why sender may not consult target now, as (error type, message), or None."""
    if target == sender or target in chain:
        return 'CycleDetected', f'{target} is already waiting in this chain of consults'
    if len(chain) + 1 >= MAX_CONSULT_DEPTH:
        return 'ConsultRefused', 'the chain of consults is too deep'
    if not policy.can_peer(sender, target):
        return 'ConsultRefused', f'{sender} may not consult {target}'
    if target not in live_peers(directory, policy, sender):
        return 'ConsultRefused', f'{target} is not available for requests'
    return None


def consult_envelope(event: Envelope, task: TaskRecord, sender: str, target: str, value: JSONValue,
                     child_id: str, budget: ModelBudget | None, now) -> Envelope:
    payload = {'input': value, 'ancestry': ancestry(event) + [sender],
               'budget': None if budget is None else budget.model_dump(mode='json')}
    return Envelope(event_id=str(uuid5(UUID(child_id), 'consult')), network_id=event.network_id, kind='consult',
        sender=sender, recipient=target, topic=None, task_id=child_id, parent_task_id=task.task_id,
        correlation_id=None, session_id=str(uuid5(UUID(child_id), 'session')), created_at=now,
        deadline=event.deadline, traceparent=event.traceparent, payload=payload, expires_at=event.expires_at)


def _pause(scope) -> float | None:
    """Stop the handler's own time limit while it waits; the deadline still bounds the wait."""
    if scope is None or scope.when() is None:
        return None
    left = scope.when() - asyncio.get_running_loop().time()
    scope.reschedule(None)
    return left


def _resume(scope, left: float | None) -> None:
    if scope is not None and left is not None:
        scope.reschedule(asyncio.get_running_loop().time() + max(left, 0.001))


def child_result(child: TaskRecord) -> TaskResult:
    return TaskResult(task_id=child.task_id, output=child.result, responding_agent=child.agent_id,
                      status=child.status, error=public_error(child.error))


class Coordinator:
    """Consults of one Worker: the Python handler path and replies that resume waiting loops."""

    def __init__(self, worker) -> None:
        self.worker = worker

    async def consult(self, ctx, target: str, value: JSONValue) -> TaskResult:
        """Ask target and wait for its answer; the handler's slot is free meanwhile.

        The handler's Python stack is not durable: after a crash, a retry-safe
        handler runs again from the start and finds the same child, otherwise
        the task becomes outcome_unknown.
        """
        async with ctx._consult_lock:  # concurrent consults of one handler run one after another
            return await self._consult(ctx, validate_name(target), value)

    async def _consult(self, ctx, target: str, value: JSONValue) -> TaskResult:
        w = self.worker
        sender = w.definition.name
        refused = refusal(w.directory, w.policy, sender, target, ancestry(ctx.event))
        if refused is not None:
            raise (CycleDetected if refused[0] == 'CycleDetected' else ConsultRefused)(refused[1])
        ctx._consults += 1
        child_id = str(uuid5(UUID(ctx.task_id), f'consult:{ctx._consults}'))
        envelope = consult_envelope(ctx.event, ctx._task, sender, target, value, child_id,
                                    inherited_budget(ctx.event) or w.definition.budget, w.clock.now())
        await w.store.wait_for_child(ctx._lease, envelope)
        paused = _pause(ctx._invocation_scope)
        ctx._slots.release()
        child = await self._await_child(child_id, ctx.event.deadline)
        await w.store.resume_waiting(ctx._lease)
        await ctx._slots.reacquire()
        _resume(ctx._invocation_scope, paused)
        if child is None:
            return TaskResult(task_id=child_id, output=None, responding_agent=target, status='expired')
        return child_result(child)

    async def _await_child(self, child_id: str, deadline) -> TaskRecord | None:
        store, clock = self.worker.store, self.worker.clock
        while clock.now() < deadline:
            child = await store.get_task(child_id)
            if child is not None and child.status in _TERMINAL:
                return child
            await asyncio.sleep(CHILD_POLL_SECONDS)
        return None

    async def accept_reply(self, delivery) -> str | None:
        """A child's reply: confirm it, and resume the waiting loop once; return that parent's task_id."""
        w, event = self.worker, delivery.envelope
        parent_id = event.parent_task_id
        if event.recipient != w.definition.name or parent_id is None or event.task_id is None:
            await self._ack(delivery)
            return None
        receipt = Envelope(event_id=str(uuid5(UUID(event.event_id), 'receipt:' + w.definition.name)),
            network_id=event.network_id, kind='receipt', sender=w.definition.name, recipient=event.sender,
            topic=None, task_id=event.task_id, parent_task_id=parent_id, correlation_id=event.event_id,
            session_id=event.session_id, created_at=w.clock.now(), deadline=event.deadline, traceparent=None,
            payload={'event_id': event.event_id, 'recipient': w.definition.name})
        if event.correlation_id is not None:
            try:
                await w.store.confirm_receipt(event.correlation_id, event.sender)  # the reply proves the request arrived
            except ReceiptNotExpected:
                pass
        resumed = await w.store.resume_parent(parent_id, event.task_id, receipt=receipt)
        await self._ack(delivery)
        return parent_id if resumed else None

    async def resume_finished(self, task: TaskRecord) -> bool:
        """Recovery: a waiting loop whose children all ended resumes even if their replies were lost."""
        waiting = task.waiting_for if isinstance(task.waiting_for, dict) else {}
        children = waiting.get('children', [])
        if waiting.get('kind') != 'loop' or not children:
            return False
        return await self.worker.store.resume_parent(task.task_id, children[0])

    async def _ack(self, delivery) -> None:
        try:
            await self.worker.bus.ack(delivery)
        except StaleDelivery:
            pass


__all__ = ['CONSULT_TOOL', 'Coordinator', 'ancestry', 'child_result', 'consult_envelope', 'consult_tool',
           'inherited_budget', 'live_peers', 'refusal', 'request_input']
