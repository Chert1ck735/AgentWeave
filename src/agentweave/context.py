"""One opaque Python invocation and its transaction-buffered publications."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
from .messages import Envelope, JSONValue, Message, encode, validate_event_topic, validate_name


@dataclass(frozen=True)
class Request:
    input: JSONValue
    history: list[Message]


class Context:
    def __init__(self, task, event, *, state=None, history=None, clock):
        self.task_id, self.session_id = task.task_id, task.session_id
        self.event = event
        self.state = {} if state is None else state
        self.history = [] if history is None else history
        self._task, self._clock, self._outgoing = task, clock, []
        self._active, self._lease_valid = True, True
        # Set by the runtime for a request handler: what consult needs to wait durably.
        self._coordinator = self._lease = self._slots = self._invocation_scope = None
        self._consults = 0
        self._consult_lock = asyncio.Lock()  # one consult at a time per handler invocation

    async def consult(self, agent: str, input: JSONValue):
        """Ask another agent of this network and wait for its TaskResult.

        The peer gets only this input, in a new session. While waiting, this
        handler's execution slot is free; its Python stack is not durable, so a
        crash reruns a retry-safe handler from the start (finding the same
        answer) and otherwise leaves the task outcome_unknown. Raises
        ConsultRefused (or CycleDetected) before sending when the peer is not
        allowed, not available, or already waiting above this task.
        """
        if not self._active or self._coordinator is None:
            raise RuntimeError('consult is available only while a request handler runs')
        self._assert_fresh()
        return await self._coordinator.consult(self, agent, input)

    def handoff(self, agent: str, *, message: str | None = None, scope: str | None = None):
        """Hand this task to agent and end the handler here; code after it does not run.

        scope 'conversation' (the default inside a conversation) also makes agent
        the conversation's active agent; 'task' moves only this task. The
        recipient gets this conversation's history and message, nothing else.
        A refused handoff fails the task with HandoffRefused.
        """
        from ._runtime.conversations import HandoffSignal
        if not self._active or self._coordinator is None:
            raise RuntimeError('handoff is available only while a request handler runs')
        if scope not in (None, 'task', 'conversation'):
            raise ValueError('scope must be task or conversation')
        raise HandoffSignal(validate_name(agent), message, scope)

    def _assert_fresh(self):
        from .errors import InputExpired, LeaseLost
        if not self._lease_valid:
            raise LeaseLost('handler lease renewal failed')
        now = self._clock.now()
        if now >= self.event.deadline or (self.event.expires_at and now >= self.event.expires_at):
            raise InputExpired('input has expired; no new effect allowed')

    async def publish(self, topic: str, payload: JSONValue, *, key: str | None = None,
                      event_time: datetime | None = None, expires_at: datetime | None = None,
                      source_version: int | None = None) -> None:
        topic = validate_event_topic(topic)
        if not self._active:
            raise RuntimeError('handler invocation has ended')
        self._assert_fresh()
        now = self._clock.now()
        envelope = Envelope(event_id=str(uuid4()), network_id=self.event.network_id,
            kind=topic, sender=self._task.agent_id, recipient=None, topic=topic,
            task_id=None, parent_task_id=self.task_id, correlation_id=self.event.event_id,
            session_id=self.session_id, created_at=now, deadline=self.event.deadline,
            traceparent=self.event.traceparent, payload=payload, key=key,
            event_time=event_time, expires_at=expires_at, source_version=source_version)
        encode(envelope)
        self._outgoing.append(envelope)


class ToolContext:
    """Runtime facts passed to a tool that annotates a parameter with this type.

    It is never part of the tool's model-facing schema. A tool can check
    attempt to tell a retry of a retry-safe call from its first attempt.
    requested_arguments are the arguments originally proposed, before any
    before_tool hook changed them; call.arguments in after hooks are the
    arguments that actually ran.
    """
    def __init__(self, call, *, task_id: str, session_id: str | None, attempt: int, clock,
                 requested_arguments: dict | None = None):
        self.tool_call_id, self.tool_id, self.caller = call.tool_call_id, call.tool_id, call.caller
        self.requested_arguments = dict(call.arguments if requested_arguments is None else requested_arguments)
        self.deadline, self.expires_at = call.deadline, call.expires_at
        self.task_id, self.session_id, self.attempt = task_id, session_id, attempt
        self._clock = clock
        self._active, self._lease_valid = True, True

    def _assert_fresh(self):
        from .errors import InputExpired, LeaseLost
        if not self._lease_valid:
            raise LeaseLost('tool lease is no longer valid')
        now = self._clock.now()
        if now >= self.deadline or (self.expires_at and now >= self.expires_at):
            raise InputExpired('input has expired; no new tool effect allowed')
