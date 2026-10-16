"""A Python worker owns bounded subscriptions and leased journal recovery."""
import asyncio
from collections import deque
import os
from uuid import uuid4
from ._records import EventRejection, Participant, SystemClock
from .messages import Envelope
from .config import _resolve_agent
from .definitions import fingerprint, SubscriptionPolicy
from .errors import DefinitionError, StaleDelivery
from ._policy import Policy
from ._runtime.delivery import AdmissionService, OutboxPump, confirm_receipts
from ._runtime.engine import RuntimeEngine
from ._runtime.handlers import SubscriptionQueue, snapshot
from ._runtime.logging import RuntimeMetrics
from ._runtime.recovery import recover

# Capacity waits back off so a blocked subscription doesn't hammer the journal.
CAPACITY_BACKOFF_SECONDS = (0.02, 0.5)
COMPATIBILITY_POLL_SECONDS = 0.02


def _managed_loop(ctx, request):
    """Marker callable: requests to an agent with a model run the managed loop."""
    raise RuntimeError('the managed loop is executed by RuntimeEngine, not called directly')


class Worker:
    def __init__(self, definition, config, store, bus, directory) -> None:
        self.definition = _resolve_agent(definition, config.root)
        model = self.definition.model
        if model is not None and self.definition.request_handler is not None:
            raise DefinitionError('an agent with a model answers requests through the managed loop; '
                                  'remove its on_request handler')
        if not self.definition.handlers and not self.definition.request_handler and model is None:
            raise DefinitionError('Worker needs a model, an on_request or an event handler')
        self.config, self.store, self.bus, self.directory = config, store, bus, directory
        self.clock = getattr(store, 'clock', SystemClock())
        self.instance_id, self.fingerprint = str(uuid4()), fingerprint(self.definition)
        self.policy = Policy([self.definition])
        self.metrics = RuntimeMetrics()
        self.recovery_wake = asyncio.Event()  # set when journaled work may have become runnable
        self.adapter = self.tools = self.model_handler = None
        if model is not None:
            from .definitions import RetryPolicy
            from .definitions.agent import HandlerDefinition
            from .providers import create_adapter
            from ._runtime.tools import ToolExecutor
            # Secrets are read by reference from the environment, never from config.
            self.adapter = create_adapter(model, os.environ)
            self.tools = ToolExecutor(self.definition, store, instance_id=self.instance_id, clock=self.clock,
                                      owner=(config.network_id, self.definition.name, '$tools'))
            # The loop replays its journal after a crash, so a restart may run it again.
            self.model_handler = HandlerDefinition(callable=_managed_loop, retry=RetryPolicy(safe=True))
        from ._runtime.coordination import Coordinator
        self.coordinator = Coordinator(self)
        self.engine = RuntimeEngine(self)
        self.pump = OutboxPump(store, bus, self.instance_id, clock=self.clock)
        self.stopping = False
        self.ready = asyncio.Event()
        self._jobs, self._readers, self._subs, self.queues = [], [], [], {}
        self._done = asyncio.Event()
        self._processing = 0
        self._runner = None
        self._announcement_attempted = False
        self._announcement_managed = False
        self._startup_jobs = {}
        self._startup_unreleased = {}
        self._shutdown_task = None
        self._shutdown_deadline = None
        self._cleanup_error = None
        self._cleanup_jobs = set()
        self._cleanup_errors = deque(maxlen=64)
        self._stop_signal = asyncio.Event()

    def handler(self, event):
        if event.kind in ('request', 'consult'):
            return self.definition.request_handler or self.model_handler
        if event.kind in ('receipt','reply','progress','directory','error','tool_call'):
            return None
        return self.definition.handlers.get(event.topic or event.kind)

    def handler_for_requests(self):
        return self.definition.request_handler or self.model_handler

    def wake_recovery(self) -> None:
        """Ask recovery to look at the journal now instead of at its next fallback poll."""
        self.recovery_wake.set()

    def compatible(self, event: Envelope) -> bool:
        self.directory.check_health()
        if event.network_id != self.config.network_id or (event.recipient and event.recipient != self.definition.name):
            return False
        capability = 'request' if event.kind in ('request', 'consult') else event.topic or event.kind
        return any(p.kind == 'agent' and p.fingerprint == self.fingerprint and capability in p.capabilities
                   and p.configuration_version == (self.definition.revision or self.fingerprint)
                   for p in self.directory.snapshot().get(self.definition.name, ()))

    async def run(self) -> None:
        if self._jobs or self.stopping:
            raise RuntimeError('Worker can only run once')
        self._runner = asyncio.current_task()
        run_error = None
        try:
            await self._startup(self.directory.start())
            participant = Participant(network_id=self.config.network_id, logical_id=self.definition.name,
                instance_id=self.instance_id, kind='agent', fingerprint=self.fingerprint,
                configuration_version=self.definition.revision or self.fingerprint,
                broker_epoch=self.config.broker_epoch or '', heartbeat_at=self.clock.now(),
                capabilities=list(self.definition.handlers) + (['request'] if self.handler_for_requests() else []))
            self._announcement_attempted = True
            self._announcement_managed = True
            await self._startup(self.directory.publish(participant),
                compensate=lambda _: self.directory.withdraw(self.definition.name, self.instance_id),
                compensate_failure=True)
            self._announcement_managed = False
            receipt_sub = await self._startup(
                self.bus.subscribe([self.bus.control_topic('receipt')], 'receipt.'+self.instance_id),
                compensate=lambda sub: sub.aclose())
            self._subs.append(receipt_sub)
            self._jobs.append(asyncio.create_task(self._controls(receipt_sub)))
            policies = {name:handler.policy for name,handler in self.definition.handlers.items()}
            policies['$inbox'] = SubscriptionPolicy(concurrency=self.definition.concurrency)
            for name, policy in policies.items():
                topic = self.bus.agent_topic(self.definition.name) if name == '$inbox' else self.bus.events_topic(name)
                sub = await self._startup(self.bus.subscribe([topic], 'worker.'+self.definition.name+'.'+name,
                                                             low_latency=name == '$inbox'),
                    compensate=lambda sub: sub.aclose())
                self._subs.append(sub)
                queue = SubscriptionQueue(policy, self.clock, self.metrics, name)
                self.queues[name] = queue
                reader = asyncio.create_task(self._read(sub, queue))
                self._jobs.append(reader)
                self._readers.append(reader)
                for _ in range(policy.concurrency):
                    self._jobs.append(asyncio.create_task(self._consume(queue)))
            self._jobs += [asyncio.create_task(self.pump.run()), asyncio.create_task(recover(self))]
            self._jobs.append(asyncio.create_task(self._stop_signal.wait()))
            self.ready.set()
            pending = set(self._jobs)
            while not self.stopping:
                completed, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for job in completed:
                    if job.cancelled() and (self.stopping or job in self._readers):
                        continue
                    error = job.exception()
                    if error is not None:
                        raise error
                    if not self.stopping:
                        raise RuntimeError('runtime service unexpectedly stopped')
        except BaseException as error:
            run_error = error
        finally:
            try:
                await asyncio.shield(self._begin_shutdown())
            except BaseException as cleanup_error:
                if run_error is not None and run_error is not cleanup_error:
                    raise BaseExceptionGroup('worker execution and cleanup failed', [run_error, cleanup_error])
                raise
        if run_error is not None:
            raise run_error

    async def _startup(self, operation, *, compensate=None, compensate_failure=False):
        """Keep acquisition owned until a synchronous handoff or compensation."""
        if self.stopping:
            operation.close()
            raise asyncio.CancelledError
        result = asyncio.get_running_loop().create_future()
        handoff = asyncio.Event()
        accepted = False
        started = False

        async def acquire():
            nonlocal started
            started = True
            self._startup_jobs[asyncio.current_task()] = 'acquiring'
            value, acquired = None, False
            try:
                value = await operation
                acquired = True
                self._startup_jobs[asyncio.current_task()] = 'handoff'
                result.set_result(value)
                await handoff.wait()
            except BaseException as error:
                if not result.done():
                    if isinstance(error, asyncio.CancelledError):
                        result.cancel()
                    else:
                        result.set_exception(error)
                raise
            finally:
                if not accepted and compensate is not None and (acquired or compensate_failure):
                    self._startup_jobs[asyncio.current_task()] = 'compensating'
                    cleanup = asyncio.create_task(compensate(value), name='aw-startup-compensation')
                    self._startup_unreleased[cleanup] = value
                    def released(done):
                        if not done.cancelled() and done.exception() is None:
                            self._startup_unreleased.pop(done, None)
                    cleanup.add_done_callback(released)
                    self._track_cleanup(cleanup)
                    now = asyncio.get_running_loop().time()
                    deadline = self._shutdown_budget()
                    # A late acquisition gets one bounded background window;
                    # this never extends the public stop/drain deadline.
                    if now > deadline:
                        deadline = now + 30
                    completed, pending = await asyncio.wait([cleanup], timeout=max(0, deadline-now))
                    if pending:
                        cleanup.cancel()
                        raise TimeoutError('startup ownership compensation deadline exhausted')
                    await cleanup

        job = asyncio.create_task(acquire(), name='aw-startup-acquisition')
        self._startup_jobs[job] = 'queued'
        self._track_cleanup(job)
        def finished(done):
            self._startup_jobs.pop(done, None)
            if not started:
                operation.close()
                result.cancel()
        job.add_done_callback(finished)
        try:
            value = await asyncio.shield(result)
            if self.stopping:
                raise asyncio.CancelledError
            accepted = True
            return value
        finally:
            # Consume a possible late acquisition error after runner cancellation.
            if not accepted:
                result.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            handoff.set()

    def _shutdown_budget(self):
        if self._shutdown_deadline is None:
            self._shutdown_deadline = asyncio.get_running_loop().time() + 30
        return self._shutdown_deadline

    def _signal_stop(self):
        self.stopping = True
        self.engine.invalidate_contexts()
        self.pump.stop()
        self._stop_signal.set()
        for job in self._jobs:
            job.cancel()

    def _begin_shutdown(self):
        if self._shutdown_task is None:
            self._shutdown_budget()
            self._signal_stop()
            self._shutdown_task = asyncio.create_task(self._shutdown(), name='aw-worker-cleanup')
            for job, phase in self._startup_jobs.items():
                # Handoff is settled by the runner's finally; compensation owns
                # its deadline and must survive shutdown's cancellation request.
                if phase in ('queued', 'acquiring'):
                    job.cancel()
            if (self._runner is not None and self._runner is not asyncio.current_task()
                    and not self.ready.is_set() and not self._runner.done()):
                self._runner.cancel()
        return self._shutdown_task

    async def _close_adapter(self):
        """Close the model adapter within the shared shutdown budget, like every other cleanup."""
        closing = asyncio.create_task(self.adapter.close(), name='aw-adapter-close')
        self._track_cleanup(closing)
        remaining = max(0, self._shutdown_deadline - asyncio.get_running_loop().time())
        _, pending = await asyncio.wait([closing], timeout=remaining)
        if pending:
            raise TimeoutError('model adapter close exceeded the worker shutdown budget')

    def _track_cleanup(self, job):
        if job in self._cleanup_jobs:
            return
        self._cleanup_jobs.add(job)
        def finished(done):
            self._cleanup_jobs.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error is not None:
                    self._cleanup_errors.append(error)
        job.add_done_callback(finished)

    async def _bounded_cleanup_wait(self, jobs, *, cancel_pending=True):
        if not jobs:
            return []
        for job in jobs:
            self._track_cleanup(job)
        remaining = max(0, self._shutdown_deadline - asyncio.get_running_loop().time())
        completed, pending = await asyncio.wait(jobs, timeout=remaining)
        errors = [job.exception() for job in completed if not job.cancelled() and job.exception() is not None]
        if cancel_pending:
            for job in pending:
                job.cancel()
        if pending:
            errors.append(TimeoutError(f'worker shutdown budget exhausted with {len(pending)} unfinished cleanup operations'))
        # Uncooperative tasks stay tracked until actual completion; never await
        # their cancellation handlers beyond the shared monotonic deadline.
        return errors

    async def _shutdown(self):
        try:
            resources = [asyncio.create_task(sub.aclose()) for sub in self._subs]
            if self._announcement_attempted and not self._announcement_managed:
                resources.append(asyncio.create_task(self.directory.withdraw(self.definition.name, self.instance_id)))
            for job in resources:
                self._track_cleanup(job)
            # Already handed-off resources can close concurrently with a late
            # startup operation; that operation owns its own compensation.
            errors = await self._bounded_cleanup_wait(list(self._startup_jobs), cancel_pending=False)
            errors += await self._bounded_cleanup_wait(self._jobs)
            errors += await self._bounded_cleanup_wait(resources)
            if len(errors) == 1:
                raise errors[0]
            if errors and all(isinstance(error, TimeoutError) for error in errors):
                raise TimeoutError('worker shutdown deadline exhausted') from ExceptionGroup('unfinished cleanup operations', errors)
            if errors:
                raise BaseExceptionGroup('worker cleanup failed', errors)
        except BaseException as error:
            self._cleanup_error = error
            raise
        finally:
            try:
                self.engine.executor.close()
                if self.tools is not None:
                    self.tools.close()
                if self.adapter is not None:
                    await self._close_adapter()
            finally:
                self._done.set()

    async def _ack(self, delivery):
        """Acknowledge an input whose outcome is already decided or durably admitted.

        After a rebalance the broker redelivers the record to the current owner,
        where admission deduplicates it, so a stale position is not a failure.
        """
        try:
            await self.bus.ack(delivery)
        except StaleDelivery:
            self.metrics.outcome('stale_ack', event=delivery.envelope, instance_id=self.instance_id)

    async def _read(self, sub, queue):
        async for delivery in sub:
            dropped, reason = await queue.put(delivery)
            if dropped:
                self.metrics.outcome(reason, event=dropped.envelope, instance_id=self.instance_id)
                await self._ack(dropped)

    async def _consume(self, queue, stop=None):
        while not self.stopping and not (stop is not None and stop.is_set()):
            delivery = await self._next(queue, stop)
            if delivery is None:
                return
            self._processing += 1
            try:
                await self._consume_one(queue, delivery)
            finally:
                self._processing -= 1

    @staticmethod
    async def _next(queue, stop):
        if stop is None:
            return await queue.get()
        getting, stopped = asyncio.create_task(queue.get()), asyncio.create_task(stop.wait())
        try:
            await asyncio.wait([getting, stopped], return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopped.cancel()
            if not getting.done():
                getting.cancel()
            await asyncio.gather(getting, stopped, return_exceptions=True)
        return getting.result() if getting.done() and not getting.cancelled() else None

    def lend_consumer(self, name):
        """Read queue name with one more consumer while a handler waits for a consult.

        Returns the event that retires that consumer once the handler continues.
        """
        queue = self.queues.get(name)
        if queue is None or self.stopping:
            return None
        stop = asyncio.Event()
        job = asyncio.create_task(self._consume(queue, stop), name='aw-lent-consumer')
        self._jobs.append(job)  # stopping the worker cancels it like any reader
        job.add_done_callback(lambda done: self._jobs.remove(done) if done in self._jobs else None)
        return stop

    async def _consume_one(self, queue, delivery):
        event = delivery.envelope
        if event.kind == 'handoff':
            # A task was handed to this agent; the journal already names it the owner.
            await self._ack(delivery)
            if event.recipient == self.definition.name and event.task_id is not None:
                await self.engine.execute(event.task_id)
            return
        if event.kind == 'reply':
            # A consulted peer answered: the waiting parent continues without a new admission.
            parent = await self.coordinator.accept_reply(delivery)
            if parent is not None:
                await self.engine.execute(parent)
            return
        if event.kind == 'tool_result':
            # A remote tool's notice: the loop reads the outcome from the journal,
            # but the notice still confirms the call's delivery after a lost receipt.
            await self._confirm(AdmissionService(self.store, self.bus), delivery)
            return
        if self.handler(event) is None or (self.definition.memory == 'stateful' and event.session_id is None):
            self.metrics.outcome('rejected', event=event, instance_id=self.instance_id)
            await self._ack(delivery)
            return
        backoff, max_backoff = CAPACITY_BACKOFF_SECONDS
        while not self.stopping:
            if not self.compatible(event):
                await asyncio.sleep(COMPATIBILITY_POLL_SECONDS)
                continue
            admitted = await self.store.admit_with_policy(event, self.fingerprint,
                agent_id=self.definition.name, max_inflight=self.definition.max_inflight,
                snapshot=snapshot(event, queue.policy))
            if isinstance(admitted, EventRejection):
                if admitted.retryable and queue.policy.overflow == 'wait':
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    continue
                self.metrics.outcome(admitted.reason, event=event, instance_id=self.instance_id)
                await self._ack(delivery)
                break
            await self._ack(delivery)
            await self.engine.execute(admitted.task_id)
            break

    async def _controls(self, sub):
        def counted(outcome, delivery):
            if outcome is not None:
                self.metrics.outcome(outcome, event=delivery.envelope, instance_id=self.instance_id)
        name = self.definition.name
        await confirm_receipts(sub, AdmissionService(self.store, self.bus),
                               lambda event: name in (event.sender, event.recipient), counted)

    async def _confirm(self, service, delivery):
        # One malformed control record must not stop this agent's intake.
        outcome = await service.accept_confirmation(delivery)
        if outcome is not None:
            self.metrics.outcome(outcome, event=delivery.envelope, instance_id=self.instance_id)

    async def drain(self) -> None:
        """Stop intake; drain and cleanup share one monotonic 30-second budget."""
        deadline = self._shutdown_budget()
        for reader in self._readers:
            reader.cancel()
        try:
            async with asyncio.timeout_at(deadline):
                while True:
                    page = await self.store.recoverable_page(self.definition.name, limit=64)
                    if any(task.status in ('outcome_unknown','waiting_peer','waiting_tool') for task in page):
                        raise RuntimeError('drain blocked by unresolved task outcome or dependency')
                    pending = await self.store.pending_runtime_count(self.definition.name)
                    if not pending and not self._processing and all(not q.items for q in self.queues.values()):
                        break
                    await asyncio.sleep(.02)
        finally:
            await self.stop()

    async def stop(self) -> None:
        await asyncio.shield(self._begin_shutdown())
