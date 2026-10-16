"""Fenced execution of one opaque handler with atomic state/result/outbox."""
import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import logging
from typing import Any
from uuid import uuid4
from pydantic import TypeAdapter
from ..context import Context, Request
from .._records import Lease, StepCommit, TaskRecord
from ..errors import InputExpired, InvalidEnvelope, LeaseLost, TaskCancelled
from ..messages import Envelope, JSONValue, Message, TaskResult, encode, error_payload, public_error
from ..errors import HandoffRefused
from .conversations import HandoffSignal, capable, hand_off, handoff_context, transferred_budget, transferred_messages
from .coordination import ancestry, inherited_budget, request_input
from .delivery import complete_or_undo
from .handlers import HandlerExecutor, HandlerNotStarted, HandlerTimeout

LEASE_TTL_SECONDS = 30
LEASE_RENEW_SECONDS = 10
_logger = logging.getLogger('agentweave.runtime')
_json_value = TypeAdapter(JSONValue)
_json_object = TypeAdapter(dict[str, JSONValue])


def reply_envelope(task: TaskRecord, event: Envelope, now: datetime, *, status: str, output: Any = None,
                   error: Any = None, safe: bool = False) -> Envelope:
    """The reply a finished task sends its requester; peers learn only the error type."""
    result = TaskResult(task_id=task.task_id, output=output, responding_agent=task.agent_id, status=status,
                        error=public_error(error))
    return Envelope(event_id=str(uuid4()), network_id=event.network_id,
        kind='reply', sender=task.agent_id, recipient=event.sender, topic=None,
        task_id=task.task_id, parent_task_id=event.parent_task_id,
        correlation_id=event.event_id, session_id=event.session_id,
        created_at=now, deadline=event.deadline,
        traceparent=None if safe else event.traceparent, payload=result.model_dump(mode='json'))


class _SlotHold:
    """Execution slots held by one attempt; a handler awaiting a consult lends them out meanwhile."""

    def __init__(self, semaphores, lend=None):
        self.semaphores, self.held, self.lend, self.lent = semaphores, False, lend, None

    async def __aenter__(self):
        await self.reacquire()
        return self

    async def __aexit__(self, *exc):
        if self.lent is not None:
            lent, self.lent = self.lent, None
            lent.abandon()  # cancelled while lent: the pool place is still counted until the task ends
        self._release_slots()

    def _release_slots(self) -> None:
        if self.held:
            self.held = False
            for semaphore in reversed(self.semaphores):
                semaphore.release()

    def release(self) -> None:
        """Lend everything this attempt occupies while it awaits a consult."""
        self._release_slots()
        if self.lend is not None and self.lent is None:
            self.lent = self.lend()

    async def reacquire(self) -> None:
        if self.held:
            return
        if self.lent is not None:
            await self.lent.restore()
            self.lent = None  # only once restored: a failed restore is undone by __aexit__
        acquired = []
        try:
            for semaphore in self.semaphores:
                await semaphore.acquire()
                acquired.append(semaphore)
        except BaseException:
            for semaphore in reversed(acquired):
                semaphore.release()
            raise
        self.held = True


class _Lent:
    """A queue reader and a pool place lent out by an attempt awaiting a consult."""

    def __init__(self, stop, executor):
        self.stop, self.executor = stop, executor
        executor.lend()

    async def restore(self) -> None:
        if self.stop is not None:
            self.stop.set()
        await self.executor.reclaim()

    def abandon(self) -> None:
        if self.stop is not None:
            self.stop.set()
        self.executor.retake()


@dataclass(frozen=True)
class _Claim:
    """Ownership of one runnable task for the duration of a single attempt."""
    task: TaskRecord
    event: Envelope
    handler: Any
    lease: Lease
    slots: Any = None


@dataclass(frozen=True)
class _Outcome:
    """What one invocation produced, before it is committed to the journal."""
    status: str = 'succeeded'
    output: Any = None
    error: Any = None
    retry: bool = False
    lease_lost: bool = False
    state: dict | None = None
    outgoing: tuple[Envelope, ...] = ()
    history: tuple[Message, ...] | None = None  # a managed loop's transcript for session memory


_LEASE_LOST = _Outcome(lease_lost=True)


class RuntimeEngine:
    def __init__(self, worker):
        self.worker = worker
        self.contexts: set[Context] = set()
        self.slots = asyncio.Semaphore(worker.definition.concurrency)
        self.blocked = False  # some task could not start; wake recovery when an attempt here ends
        self.executor = HandlerExecutor(worker.definition.concurrency,
            owner=(worker.config.network_id, worker.definition.name))
        self.subscription_slots = {name:asyncio.Semaphore(handler.policy.concurrency)
                                   for name, handler in worker.definition.handlers.items()}
        self.subscription_slots['request'] = asyncio.Semaphore(worker.definition.concurrency)

    async def execute(self, task_id: str) -> None:
        """Run one attempt of an admitted task if this instance can own it.

        A task that could not start because a limit or gate was busy waits in the
        journal; when an attempt here ends, recovery is woken to start it.
        """
        attempted = await self._execute(task_id)
        if attempted and self.blocked:
            self.blocked = False
            self.worker.wake_recovery()

    async def _execute(self, task_id: str) -> bool:
        w = self.worker
        task = await w.store.expire_task(task_id)
        if not task or task.status not in ('accepted','running'):
            return False
        event = await w.store.get_input_envelope(task_id)
        handler = w.handler(event)
        if handler is None:
            return False
        name = 'request' if event.kind in ('request', 'consult') else event.topic or event.kind
        queue = '$inbox' if name == 'request' else name
        async with _SlotHold([self.subscription_slots[name], self.slots],
                             lend=lambda: self._lend(queue)) as slots:
            claim = await self._acquire(task, event, handler, name, slots)
            if claim is None:
                return False
            outcome = await self._invoke(claim)
            if outcome.lease_lost:
                return True
            if outcome.status == 'handoff':
                await self._hand_off(claim, *outcome.output)
                return True
            if outcome.retry:
                # The delay is journaled, so the slot is freed now and no replica retries early.
                retry_at = w.clock.now() + timedelta(seconds=handler.retry.min_delay)
                await self._release(claim.lease, error=outcome.error, retry_at=retry_at)
                asyncio.get_running_loop().call_later(handler.retry.min_delay, w.wake_recovery)
                return True
            await self._commit(claim, outcome)
            return True

    def _lend(self, queue: str) -> '_Lent':
        """Lend what an attempt awaiting a consult occupies: a queue reader and a pool place."""
        return _Lent(self.worker.lend_consumer(queue), self.executor)

    async def _acquire(self, task: TaskRecord, event: Envelope, handler: Any, name: str, slots) -> _Claim | None:
        """Lease the task, take its ordering gates and record the attempt start."""
        w = self.worker
        conversation = (await w.store.get_conversation(task.session_id)
                        if task.session_id is not None and event.kind == 'request' else None)
        if conversation is None:
            return await self._begin(task, event, handler, name, slots)
        lease = await w.store.claim_runtime(task.task_id, w.instance_id, LEASE_TTL_SECONDS,
                                            concurrency=w.definition.concurrency, agent_id=w.definition.name)
        if lease is None:
            self.blocked = True
            return None
        # Read again under the lease: a handoff may have changed the task since it was picked.
        task, event = await w.store.get_task(task.task_id), await w.store.get_input_envelope(task.task_id)
        conversation = (await w.store.get_conversation(task.session_id)
                        if task.session_id is not None and event.kind == 'request' else None)
        gates = []
        if w.definition.memory == 'stateful' or conversation is not None:
            # Messages of one conversation run one after another, whoever answers them.
            gates.append(('$session', task.session_id))
        if handler.policy.ordering == 'key':
            gates.append((name, event.key or event.session_id or event.event_id))
        for subscription, key in gates:
            if not await w.store.acquire_key_gate(task.task_id, subscription, key, lease=lease):
                await self._release(lease)
                self.blocked = True  # an earlier message of this conversation runs first
                return None
        if not w.compatible(event):
            await self._release(lease)
            return None
        if conversation is not None and task.transfer_context is None:
            # Read again under the session gate: a handoff may have just moved the conversation.
            conversation = await w.store.get_conversation(task.session_id)
            if conversation.active_agent != w.definition.name and capable(w.directory, conversation.active_agent, event):
                # Queued before a handoff: the conversation's current owner answers it.
                try:
                    await w.store.transfer(lease, conversation.active_agent, {'messages': []}, 'task', counted=False)
                    return None
                except LeaseLost:
                    return None
                except HandoffRefused:
                    pass  # the owner cannot take it now: answer it here rather than strand it
        if not await w.store.start_handler(lease, retry_safe=handler.retry.safe,
                                           max_attempts=handler.retry.max_attempts):
            current = await w.store.get_task(task.task_id)
            if current.status == 'outcome_unknown':
                w.metrics.outcome('unknown_outcomes', event=event, task_id=task.task_id, instance_id=w.instance_id)
            return None
        return _Claim(task=await w.store.get_task(task.task_id), event=event, handler=handler, lease=lease, slots=slots)

    async def _begin(self, task: TaskRecord, event: Envelope, handler: Any, name: str, slots) -> _Claim | None:
        """Claim, gates and attempt start in one journal transaction (a task outside any conversation)."""
        w = self.worker
        if not w.compatible(event):
            return None
        gates = [('$session', task.session_id)] if w.definition.memory == 'stateful' else []
        if handler.policy.ordering == 'key':
            gates.append((name, event.key or event.session_id or event.event_id))
        # A cancelled worker still learns whether the attempt began, and takes it back if it did.
        lease = await complete_or_undo(w.store.begin_attempt(task.task_id, w.instance_id, LEASE_TTL_SECONDS,
                                                             concurrency=w.definition.concurrency,
                                                             agent_id=w.definition.name, gates=gates,
                                                             retry_safe=handler.retry.safe,
                                                             max_attempts=handler.retry.max_attempts),
                                       w.store.abandon_attempt)
        if lease is None:
            self.blocked = True  # a busy limit or gate frees later: recovery is woken then
            current = await w.store.get_task(task.task_id)
            if current is not None and current.status == 'outcome_unknown':
                w.metrics.outcome('unknown_outcomes', event=event, task_id=task.task_id, instance_id=w.instance_id)
            return None
        # Read again under the lease: a handoff may have changed the task since it was picked.
        task, event = await w.store.get_task(task.task_id), await w.store.get_input_envelope(task.task_id)
        return _Claim(task=task, event=event, handler=handler, lease=lease, slots=slots)

    async def _invoke(self, claim: _Claim) -> _Outcome:
        """Call the opaque handler once while a background task renews the lease."""
        w, task, event, handler = self.worker, claim.task, claim.event, claim.handler
        if event.kind == 'consult' and w.definition.name in ancestry(event):
            return _Outcome(status='failed', error={'type': 'CycleDetected',
                                                    'message': 'this agent is already waiting in the chain'})
        if handler is w.model_handler:
            return await self._invoke_model(claim)
        state, history = {}, []
        if w.definition.memory == 'stateful':
            state, history = await w.store.load_session(task.agent_id, task.session_id)
        history = history + transferred_messages(task)  # what a previous owner handed over
        ctx = Context(task, event, state=state, history=history, clock=w.clock)
        self.contexts.add(ctx)
        requested = event.kind in ('request', 'consult')
        if requested:
            ctx._coordinator, ctx._lease, ctx._slots = w.coordinator, claim.lease, claim.slots
        argument = Request(request_input(event, task.input), history.copy()) if requested else event.payload
        renewer = asyncio.create_task(self._renew(claim.lease, ctx))
        w.metrics.outcome('attempts', event=event, task_id=task.task_id, instance_id=w.instance_id)
        outcome, invocation = _Outcome(), None
        try:
            invocation = asyncio.create_task(self.executor.invoke(handler.callable, ctx, argument,
                min(handler.retry.timeout, max(.001, self._remaining(event)))))
            completed, _ = await asyncio.wait([invocation, renewer], return_when=asyncio.FIRST_COMPLETED)
            if renewer in completed:
                ctx._active = ctx._lease_valid = False
                invocation.cancel()
                await asyncio.gather(invocation, return_exceptions=True)
                return _LEASE_LOST
            outcome = _Outcome(output=invocation.result())
        except (HandlerNotStarted, InputExpired):
            outcome = _Outcome(status='expired')
        except LeaseLost:
            return _LEASE_LOST
        except HandoffSignal as signal:
            outcome = _Outcome(status='handoff', output=(signal, list(ctx.history)))
        except HandlerTimeout:
            outcome = _Outcome(status='outcome_unknown')
        except asyncio.CancelledError:
            # Stop/crash doesn't claim a resumable coroutine or commit partial state.
            raise
        except Exception as error:
            outcome = self._failure(claim, error)
        finally:
            ctx._active = False
            if invocation is not None and not invocation.done():
                ctx._lease_valid = False
                invocation.cancel()
            renewer.cancel()
            finishing = [renewer] + ([invocation] if invocation is not None else [])
            renewed = (await asyncio.gather(*finishing, return_exceptions=True))[0]
            self.contexts.discard(ctx)
        if isinstance(renewed, Exception):
            return _LEASE_LOST
        if outcome.status == 'succeeded':
            outcome = self._settle(claim, outcome, ctx.state)
        return replace(outcome, state=ctx.state, outgoing=tuple(ctx._outgoing))

    async def _invoke_model(self, claim: _Claim) -> _Outcome:
        """Run the managed AI loop under the task lease; its journals make it replayable."""
        from .steps import ModelLoop
        w, task, event = self.worker, claim.task, claim.event
        state, history = {}, []
        if w.definition.memory == 'stateful':
            state, history = await w.store.load_session(task.agent_id, task.session_id)
        ctx = Context(task, event, state=state, history=history, clock=w.clock)

        async def progress(attempt_id: str, text: str) -> None:
            now = w.clock.now()
            await w.store.append_progress(task.task_id, Envelope(event_id=str(uuid4()),
                network_id=event.network_id, kind='progress', sender=task.agent_id, recipient=None,
                topic='progress', task_id=task.task_id, parent_task_id=event.parent_task_id,
                correlation_id=event.event_id, session_id=event.session_id, created_at=now,
                deadline=event.deadline, traceparent=event.traceparent,
                payload={'attempt_id': attempt_id, 'text': text}))
        loop = ModelLoop(w.definition, w.store, w.adapter, instance_id=w.instance_id, clock=w.clock,
                         tools=w.tools, max_model_calls=w.config.limits.max_model_calls,
                         pool_limit=w.config.limits.provider_pool_limit,
                         history=history + transferred_messages(task), progress=progress,
                         directory=w.directory, bus=w.bus, coordinator=w.coordinator)
        renewer = asyncio.create_task(self._renew(claim.lease, ctx))
        running = asyncio.create_task(loop.run(task, claim.lease))
        w.metrics.outcome('attempts', event=event, task_id=task.task_id, instance_id=w.instance_id)
        try:
            completed, _ = await asyncio.wait([running, renewer], return_when=asyncio.FIRST_COMPLETED)
            if renewer in completed:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                return _LEASE_LOST
            outcome = running.result()
            if outcome.status in ('waiting_peer', 'handed_off'):
                return _LEASE_LOST  # the wait is journaled and the lease released: nothing to commit
        except LeaseLost:
            return _LEASE_LOST
        except TaskCancelled:
            # The step that ran has reported; the next one may not start.
            return _Outcome(status='cancelled', error={'type': 'Cancelled', 'message': 'an operator cancelled the task'})
        finally:
            # A stopped or cancelled worker must not leave the loop sending requests.
            for task_ in (renewer, running):
                if not task_.done():
                    task_.cancel()
            await asyncio.gather(renewer, running, return_exceptions=True)
        if outcome.status == 'succeeded':
            settled = self._settle(claim, _Outcome(output=outcome.output), state)
            return replace(settled, state=state, history=outcome.messages)
        return _Outcome(status=outcome.status, error=outcome.error)

    async def _hand_off(self, claim: _Claim, signal: HandoffSignal, history: list) -> None:
        """Move the task as ctx.handoff asked; a refusal fails it with the owner unchanged."""
        w, task, event = self.worker, claim.task, claim.event
        budget = transferred_budget(task) or inherited_budget(event) or w.definition.budget
        try:
            await hand_off(w, claim.lease, task, event, signal.agent,
                           handoff_context(history, signal.message, w.definition.name, budget), signal.scope)
        except LeaseLost:
            return
        except HandoffRefused as error:
            await self._commit(claim, _Outcome(status='failed', error=error_payload(error)))

    async def _release(self, lease: Lease, **details) -> None:
        """Give up a claim; a lease already lost needs no release and must not stop the Worker."""
        try:
            await self.worker.store.release_claim(lease, **details)
        except LeaseLost:
            _logger.info('claim already lost before release', extra={'task_id': lease.task_id})

    def _failure(self, claim: _Claim, error: Exception) -> _Outcome:
        retry = claim.handler.retry.safe
        _logger.error('handler failed%s', '; a retry-safe attempt will run again' if retry else '',
            exc_info=error, extra={'task_id':claim.task.task_id, 'agent_id':claim.task.agent_id,
                                   'event_id':claim.event.event_id})
        return _Outcome(status='failed', error=error_payload(error), retry=retry)

    def _settle(self, claim: _Claim, outcome: _Outcome, state: dict) -> _Outcome:
        """Validate a returned value; a late success has no committed effects."""
        try:
            output = _json_value.validate_python(outcome.output)
            _json_object.validate_python(state)
        except (ValueError, TypeError) as error:
            # Validation messages quote the returned value; keep it out of the journal.
            return _Outcome(status='failed', error={'type': type(error).__name__,
                'message': 'handler result or session state is not JSON-compatible'})
        now = self.worker.clock.now()
        event = claim.event
        if event.deadline <= now or (event.expires_at and event.expires_at <= now):
            return _Outcome(status='expired')
        return replace(outcome, output=output)

    async def _commit(self, claim: _Claim, outcome: _Outcome) -> None:
        """Atomically record the result, session state, history and publications."""
        w, task, event = self.worker, claim.task, claim.event
        now = w.clock.now()
        stateful = w.definition.memory == 'stateful'
        succeeded = outcome.status == 'succeeded'
        if not (succeeded and stateful):
            messages = []
        elif outcome.history is not None:
            messages = list(outcome.history)
        else:
            messages = [Message(role='user', content=request_input(event, task.input)),
                        Message(role='assistant', content=outcome.output)]
        outgoing = list(outcome.outgoing) if succeeded else []
        # Only an addressed input's sender waits for an answer; a topic publisher waits for nothing.
        answered = event.recipient is not None
        try:
            if outcome.status != 'outcome_unknown' and answered:
                outgoing.append(self._reply(task, event, now, outcome))
            for envelope in outgoing:
                encode(envelope)
        except (InvalidEnvelope, ValueError, TypeError) as error:
            # Output framing is part of this completed invocation, not a
            # storage failure or a reason to execute opaque code again.
            outcome = _Outcome(status='failed', error={'type': type(error).__name__,
                'message': 'handler publications or reply could not be encoded'})
            succeeded, messages = False, []
            failure = self._reply(task, event, now, outcome, safe=True)
            encode(failure)
            outgoing = [failure] if answered else []
        try:
            await w.store.commit_step(claim.lease, StepCommit(step_id='handler.'+str(claim.lease.token),
                expected_version=task.version, status=outcome.status, result=outcome.output,
                state=outcome.state if succeeded and stateful else None, error=outcome.error,
                messages=messages, outgoing=outgoing))
        except LeaseLost:
            return
        w.metrics.outcome('unknown_outcomes' if outcome.status == 'outcome_unknown' else outcome.status,
            event=event, task_id=task.task_id, instance_id=w.instance_id)
        w.metrics.latency((now-(task.admitted_at or event.created_at)).total_seconds())

    def _remaining(self, event: Envelope) -> float:
        now = self.worker.clock.now()
        remaining = (event.deadline - now).total_seconds()
        if event.expires_at:
            remaining = min(remaining, (event.expires_at - now).total_seconds())
        return remaining

    def invalidate_contexts(self) -> None:
        for context in self.contexts:
            context._active = context._lease_valid = False

    @staticmethod
    def _reply(task: TaskRecord, event: Envelope, now: datetime, outcome: _Outcome, *,
               safe: bool = False) -> Envelope:
        return reply_envelope(task, event, now, status=outcome.status, output=outcome.output,
                              error=outcome.error, safe=safe)

    async def _renew(self, lease: Lease, context: Context) -> None:
        while True:
            await asyncio.sleep(LEASE_RENEW_SECONDS)
            try:
                lease = await self.worker.store.renew(lease)
            except Exception:
                context._lease_valid = False
                raise
