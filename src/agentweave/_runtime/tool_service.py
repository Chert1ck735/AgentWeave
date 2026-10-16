"""A process serving published Python tools to agents of its network.

Each tool is announced as namespace.name with its schema and revision and
reads an inbox shared by all its replicas. Calls are admitted as the tool's
own tasks and run by the same ToolExecutor that runs an agent's local tools.
"""
import asyncio
import logging
import re
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4
from pydantic import ValidationError
from .._records import EventRejection, Lease, Participant, StepCommit, SystemClock, TaskRecord
from ..definitions import fingerprint, load_object
from ..definitions.agent import Agent
from ..definitions.tool import ToolDefinition
from ..definitions.toolbox import HttpToolSpec, McpToolboxSpec, PythonBinding, ToolboxDefinition
from ..errors import DefinitionError, LeaseLost, StaleDelivery, StoreConflict
from ..messages import Envelope, TaskResult, ToolCall, ToolResult, public_error, validate_name
from ..ports import Directory, EventBus, ExecutionStore
from .tools import ToolExecutor, remote_task_id

_logger = logging.getLogger('agentweave.tools')

def default_namespace(root: Path | str) -> str:
    """The project's name as a tool namespace: its directory name in logical-name characters."""
    return validate_name(re.sub(r'[^A-Za-z0-9_-]', '_', Path(root).name).lstrip('-') or 'project')


def published_tools(definition: ToolDefinition | ToolboxDefinition, namespace: str | None,
                    root: Path | str) -> dict[str, ToolDefinition | HttpToolSpec]:
    """Each published tool ID, namespace.name, with its binding renamed to that ID.

    A single tool defaults to the project's namespace, a toolbox to its own.
    """
    if isinstance(definition, McpToolboxSpec):
        validate_name(namespace or definition.namespace)
        return {}  # MCP tools are discovered from the server while the service runs
    if isinstance(definition, ToolDefinition):
        bindings = {definition.name: definition}
    elif isinstance(definition, ToolboxDefinition):
        namespace = namespace or definition.namespace
        bindings = {}
        for name, binding in definition.bindings.items():
            if isinstance(binding, PythonBinding):
                binding = load_object(binding.target, Path(root))[0]
            if not isinstance(binding, (ToolDefinition, HttpToolSpec)):
                raise DefinitionError(f'binding {name} is not a Python or HTTP tool')
            bindings[name] = binding
    else:
        raise DefinitionError('a tool service publishes a ToolDefinition, ToolboxDefinition or McpToolboxSpec')
    namespace = validate_name(namespace or default_namespace(root))
    return {f'{namespace}.{name}': binding.model_copy(update={'name': validate_name(f'{namespace}.{name}')})
            for name, binding in bindings.items()}


class _ServiceExecutor(ToolExecutor):
    """Runs calls from other agents: the tool's allowed_callers decide, not a calling agent's policy."""

    def _may_call(self, call: ToolCall, binding: ToolDefinition) -> bool:
        return binding.allowed_callers is None or call.caller in binding.allowed_callers


# A rejected call ran nothing; the task records why, the tool journal any effect.
_TASK_STATUS = {'succeeded': 'succeeded', 'failed': 'failed', 'denied': 'failed', 'invalid': 'failed',
                'expired': 'expired', 'outcome_unknown': 'outcome_unknown'}
TOOL_MAX_INFLIGHT = 64
TOOL_SHUTDOWN_SECONDS = 30
TOOL_RECOVERY_SECONDS = 0.1
ANNOUNCE_RETRY_SECONDS = 1.0
CAPACITY_BACKOFF_SECONDS = (0.02, 0.5)


class ToolWorker:
    """Serve published Python tools from their own process.

    Each tool is announced in the live directory as namespace.name with its
    schema, revision and this instance, and reads one inbox whose consumer
    group all replicas share. A call is admitted as a task of the tool, run by
    a ToolExecutor under that task's lease and committed with its result;
    a repeated delivery of one tool_call_id finds the saved outcome instead of
    running the function again. The caller's code is never needed here, nor
    this tool's code in the caller.
    """

    def __init__(self, definition, config, store: ExecutionStore, bus: EventBus, directory: Directory, *,
                 namespace: str | None = None, concurrency: int = 1, instance_id: str | None = None) -> None:
        from .delivery import OutboxPump
        if type(concurrency) is not int or concurrency < 1:
            raise DefinitionError('tool service concurrency must be a positive integer')
        self.config, self.store, self.bus, self.directory = config, store, bus, directory
        self.bindings = published_tools(definition, namespace, config.root)
        self.mcp = None
        if isinstance(definition, McpToolboxSpec):
            from ..adapters.tools.mcp import McpBinding
            self.mcp_spec = definition if namespace is None else definition.model_copy(update={'namespace': namespace})
            self.namespace = self.mcp_spec.namespace
            self.mcp = McpBinding(self.mcp_spec, cwd=config.root, on_change=self._mcp_changed)
        else:
            self.namespace = next(iter(self.bindings)).rsplit('.', 1)[0]
        self.clock = getattr(store, 'clock', None) or SystemClock()
        self.instance_id, self.concurrency = instance_id or str(uuid4()), concurrency
        self.fingerprints = {tool_id: fingerprint(binding) for tool_id, binding in self.bindings.items()}
        service = Agent(self.namespace, tools=list(self.bindings.values()), concurrency=concurrency)
        self.executor = _ServiceExecutor(service, store, instance_id=self.instance_id, clock=self.clock,
                                         owner=(config.network_id, '$tools', self.namespace))
        self.pump = OutboxPump(store, bus, self.instance_id, clock=self.clock)
        self.stopping = False
        self.ready = asyncio.Event()
        self._stopped = asyncio.Event()
        self._slots = asyncio.Semaphore(concurrency)
        self._jobs, self._subs, self._published = [], [], []
        self._intakes: dict[str, tuple] = {}  # tool_id -> (subscription, job)
        self._unpublished: set[str] = set()  # announcements to retry, e.g. after a revision change
        self._sync_lock = asyncio.Lock()
        self._sync_dirty = False  # a catalog update failed and must run again
        self._running: dict[str, asyncio.Task] = {}
        self._failure: BaseException | None = None
        self._shutdown_task = None

    @property
    def logical_ids(self) -> tuple[str, ...]:
        return tuple(self.bindings)

    def participant(self, tool_id: str) -> Participant:
        binding = self.bindings[tool_id]
        schema = binding.schema
        return Participant(network_id=self.config.network_id, logical_id=tool_id, instance_id=self.instance_id,
            kind='tool', fingerprint=self.fingerprints[tool_id], configuration_version=schema.revision,
            broker_epoch=self.config.broker_epoch or '', heartbeat_at=self.clock.now(),
            description=schema.description, capabilities=['tool_call'], name=schema.tool_id,
            revision=schema.revision, input_schema=schema.input_schema, output_schema=schema.output_schema,
            allowed_callers=None if binding.allowed_callers is None else list(binding.allowed_callers))

    async def run(self) -> None:
        if self._jobs or self.stopping:
            raise RuntimeError('ToolWorker can only run once')
        try:
            await self.directory.start()
            await self._subscribe(self.bus.control_topic('receipt'), 'receipt.' + self.instance_id, self._receipts,
                                  name='aw-tool-receipts')
            if self.mcp is not None:
                await self.mcp.connect()
            async with self._sync_lock:  # a concurrent catalog change waits until all is open
                if self.mcp is not None:
                    await self._apply_mcp()
                for tool_id in list(self.bindings):
                    if tool_id not in self._intakes:
                        await self._open_tool(tool_id)
                self.ready.set()
            self._jobs += [asyncio.create_task(self.pump.run(), name='aw-tool-outbox'),
                           asyncio.create_task(self._recover(), name='aw-tool-recovery')]
            stopped = asyncio.create_task(self._stopped.wait())
            try:
                done, _ = await asyncio.wait([*self._jobs, stopped], return_when=asyncio.FIRST_COMPLETED)
            finally:
                stopped.cancel()
            if self._failure is not None:
                raise self._failure
            for job in done:
                if job is not stopped and not job.cancelled() and job.exception() is not None:
                    raise job.exception()
            if not self.stopping:
                raise RuntimeError('tool service unexpectedly stopped')
        finally:
            await asyncio.shield(self._begin_shutdown())

    async def stop(self) -> None:
        """Stop intake, let running calls finish within the shutdown budget, then withdraw."""
        self.stopping = True
        self._stopped.set()
        await asyncio.shield(self._begin_shutdown())

    drain = stop

    def _begin_shutdown(self):
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown(), name='aw-tool-shutdown')
        return self._shutdown_task

    async def _shutdown(self) -> None:
        self.stopping = True
        # Intake stops first; receipts and results of finishing calls still go out.
        intake = [job for job in self._jobs if job.get_name() in ('aw-tool-intake', 'aw-tool-recovery')]
        for job in intake:
            job.cancel()
        await asyncio.gather(*intake, return_exceptions=True)
        running = list(self._running.values())
        if running:
            _, unfinished = await asyncio.wait(running, timeout=TOOL_SHUTDOWN_SECONDS)
            for job in unfinished:
                job.cancel()  # a non-retry-safe call left running becomes outcome_unknown
            await asyncio.gather(*running, return_exceptions=True)
        if self.mcp is not None:
            await self.mcp.close()  # stops a stdio server this service started, never a URL server
        try:
            await self.pump.run_once()  # flush the last results; other pumps publish the rest
        except Exception as error:
            _logger.warning('final outbox flush failed', exc_info=error)
        self.pump.stop()
        for job in self._jobs:
            job.cancel()
        await asyncio.gather(*self._jobs, return_exceptions=True)
        errors = []
        for subscription in self._subs:
            try:
                await subscription.aclose()
            except Exception as error:
                errors.append(error)
        for tool_id in self._published:
            try:
                await self.directory.withdraw(tool_id, self.instance_id)
            except Exception as error:
                errors.append(error)
        self.executor.close()
        if errors:
            raise errors[0]

    async def _subscribe(self, topic: str, group: str, consume: Callable, *, name: str = 'aw-tool-intake',
                         low_latency: bool = False):
        subscription = await self.bus.subscribe([topic], group, low_latency=low_latency)
        self._subs.append(subscription)
        job = asyncio.create_task(consume(subscription), name=name)
        self._jobs.append(job)
        return subscription, job

    async def _open_tool(self, tool_id: str) -> None:
        """Announce a tool and read its inbox, shared by every replica of this tool."""
        await self._announce(tool_id)
        self._intakes[tool_id] = await self._subscribe(
            self.bus.tool_topic(tool_id), 'tool.' + tool_id, lambda subscription: self._intake(tool_id, subscription),
            low_latency=True)

    async def _announce(self, tool_id: str) -> None:
        if tool_id not in self._published:
            self._published.append(tool_id)
        try:
            await self.directory.publish(self.participant(tool_id))
        except (StoreConflict, DefinitionError) as error:
            if self.mcp is None:
                raise
            # E.g. calls of the old revision are still unfinished; announce again shortly.
            _logger.warning('tool announcement deferred: %s', error, extra={'tool_id': tool_id})
            self._unpublished.add(tool_id)
            return
        self._unpublished.discard(tool_id)

    async def _close_tool(self, tool_id: str) -> None:
        subscription, job = self._intakes.pop(tool_id, (None, None))
        if job is not None:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
            self._jobs.remove(job)
        if subscription is not None:
            self._subs.remove(subscription)
            await subscription.aclose()
        await self._withdraw(tool_id)

    async def _withdraw(self, tool_id: str) -> None:
        self._unpublished.discard(tool_id)
        if tool_id in self._published:
            await self.directory.withdraw(tool_id, self.instance_id)
            self._published.remove(tool_id)  # only once withdrawn, or shutdown withdraws it again
            # The directory folds its own withdrawal asynchronously; announce a new revision after it.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5
            while loop.time() < deadline and any(entry.instance_id == self.instance_id
                                                 for entry in self.directory.snapshot().get(tool_id, ())):
                await asyncio.sleep(0.02)

    async def _mcp_changed(self) -> None:
        try:
            await self._sync_mcp()
        except Exception as error:
            # A directory or broker problem must not tear down the MCP connection; recovery retries.
            self._sync_dirty = True
            _logger.error('MCP tool catalog update failed', exc_info=error, extra={'namespace': self.namespace})

    async def _sync_mcp(self) -> None:
        async with self._sync_lock:
            await self._apply_mcp()

    async def _apply_mcp(self) -> None:
        """Serve the server's current tools; idempotent, so a failed update can simply run again.

        The new catalog replaces the executor's tools before any announcement, so
        a call checked against a changed or removed revision is refused from then on.
        """
        from ..adapters.tools.mcp import McpTool, tool_fingerprint
        if self.stopping:
            return
        self._sync_dirty = False
        schemas = {f'{self.namespace}.{schema.tool_id}': schema for schema in await self.mcp.list_schemas()}
        current = self.bindings
        removed = [tool_id for tool_id in current if tool_id not in schemas]
        changed = {tool_id for tool_id, schema in schemas.items()
                   if tool_id in current and current[tool_id].schema.revision != schema.revision}
        tools = {tool_id: current[tool_id] if tool_id in current and tool_id not in changed else
                 McpTool(tool_id, schema.tool_id, schema, self.mcp_spec, self.mcp)
                 for tool_id, schema in schemas.items()}
        self.bindings = tools
        self.fingerprints = {tool_id: tool_fingerprint(self.mcp_spec, schemas[tool_id]) for tool_id in tools}
        self.executor.tools = dict(tools)
        for tool_id in removed:
            await self._close_tool(tool_id)
        for tool_id in changed:
            await self._withdraw(tool_id)
        if not self.ready.is_set():
            return  # run() opens everything once it is set up
        for tool_id in tools:
            if tool_id not in self._intakes:
                await self._open_tool(tool_id)
            elif tool_id in changed or tool_id in self._unpublished or tool_id not in self._published:
                await self._announce(tool_id)

    async def _ack(self, delivery) -> None:
        try:
            await self.bus.ack(delivery)
        except StaleDelivery:
            pass  # redelivered to the current owner, where admission deduplicates it

    async def _receipts(self, subscription) -> None:
        from .delivery import AdmissionService, confirm_receipts
        await confirm_receipts(subscription, AdmissionService(self.store, self.bus),
                               lambda event: event.sender in self.bindings or event.recipient in self.bindings)

    async def _intake(self, tool_id: str, subscription) -> None:
        async for delivery in subscription:
            task_id = await self._admit(tool_id, delivery)
            if task_id is not None:
                self._spawn(task_id)

    def _call(self, tool_id: str, event: Envelope) -> ToolCall | None:
        """The call an inbox envelope carries, if it is addressed and identified consistently."""
        if (event.network_id != self.config.network_id or event.kind != 'tool_call' or
                event.recipient != tool_id or event.task_id is None):
            return None
        try:
            call = ToolCall.model_validate(event.payload)
        except ValidationError:
            return None
        if call.tool_id != tool_id or call.caller != event.sender or event.task_id != remote_task_id(call.tool_call_id):
            return None
        return call

    async def _admit(self, tool_id: str, delivery) -> str | None:
        """Durably accept a call (its receipt goes out with it), or drop what cannot run."""
        event = delivery.envelope
        if self._call(tool_id, event) is None:
            _logger.warning('ignoring an invalid tool call', extra={'event_id': event.event_id, 'tool_id': tool_id})
            await self._ack(delivery)
            return None
        delay = CAPACITY_BACKOFF_SECONDS[0]
        while not self.stopping:
            try:
                admitted = await self.store.admit_with_policy(event, self.fingerprints[tool_id], agent_id=tool_id,
                                                              max_inflight=TOOL_MAX_INFLIGHT)
            except StoreConflict as error:
                # E.g. another revision owns the tool now; the caller's outbox delivers again.
                _logger.warning('tool call not admitted: %s', error, extra={'event_id': event.event_id})
                await self._ack(delivery)
                return None
            if isinstance(admitted, EventRejection):
                if admitted.retryable:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, CAPACITY_BACKOFF_SECONDS[1])
                    continue
                await self._ack(delivery)  # expired: the caller decides at its deadline
                return None
            await self._ack(delivery)
            return admitted.task_id
        return None

    def _spawn(self, task_id: str) -> None:
        if task_id in self._running or self.stopping:
            return
        job = asyncio.create_task(self._execute(task_id), name='aw-tool-call')
        self._running[task_id] = job
        job.add_done_callback(lambda done: self._finished(task_id, done))

    def _finished(self, task_id: str, job: asyncio.Task) -> None:
        self._running.pop(task_id, None)
        if job.cancelled() or job.exception() is None:
            return
        # Storage failures stop the service visibly; recovery resumes the call after a restart.
        _logger.error('tool call execution failed', exc_info=job.exception(), extra={'task_id': task_id})
        if self._failure is None:
            self._failure = job.exception()
            self._stopped.set()

    async def _recover(self) -> None:
        """Pick up admitted calls nobody runs, e.g. after another replica stopped mid-call."""
        loop = asyncio.get_running_loop()
        announce_at = 0.0
        while not self.stopping:
            if (self._unpublished or self._sync_dirty) and loop.time() >= announce_at:
                announce_at = loop.time() + ANNOUNCE_RETRY_SECONDS
                await self._retry_catalog()
            for tool_id in list(self.fingerprints):
                cursor = None
                while page := await self.store.recoverable_page(tool_id, after_task_id=cursor, limit=64):
                    for task in page:
                        # A call of an older revision is spawned too, to be refused at once.
                        if task.status != 'outcome_unknown':
                            self._spawn(task.task_id)
                    cursor = page[-1].task_id
            await asyncio.sleep(TOOL_RECOVERY_SECONDS)

    async def _retry_catalog(self) -> None:
        """Run a failed catalog update again and repeat deferred announcements; never fatal."""
        try:
            async with self._sync_lock:
                if self._sync_dirty and self.mcp is not None:
                    await self._apply_mcp()
                for tool_id in [tool_id for tool_id in self._unpublished if tool_id in self.bindings]:
                    await self._announce(tool_id)
        except Exception as error:
            self._sync_dirty = self.mcp is not None
            _logger.warning('tool catalog retry failed: %s', type(error).__name__, extra={'namespace': self.namespace})

    def _superseded(self, task: TaskRecord) -> bool:
        """No live instance serves the revision this task was admitted for any more."""
        return all(entry.fingerprint != task.definition_hash
                   for entry in self.directory.snapshot().get(task.agent_id, ()))

    async def _execute(self, task_id: str) -> None:
        from .engine import LEASE_TTL_SECONDS
        async with self._slots:
            task = await self.store.expire_task(task_id)
            if task is None or task.status not in ('accepted', 'running'):
                return
            expected = self.fingerprints.get(task.agent_id)
            if expected is None:
                return
            if task.definition_hash != expected and not self._superseded(task):
                return  # another replica still serves that revision
            lease = await self.store.claim_runtime(task_id, self.instance_id, LEASE_TTL_SECONDS,
                                                   concurrency=self.concurrency)
            if lease is None:
                return
            event = await self.store.get_input_envelope(task_id)
            task = await self.store.get_task(task_id)
            if task.definition_hash != expected:
                # Admitted for a revision no longer served: the caller learns before any effect.
                await self.store.commit_step(lease, StepCommit(step_id='tool.' + str(lease.token),
                    expected_version=task.version, status='failed', error={'type': 'ToolRevisionChanged',
                    'message': 'tool schema changed; refresh and call again'}))
                return
            call = self._call(task.agent_id, event)
            if call is None:
                # Never a reason to stop every replica: the task fails, nothing ran.
                await self.store.commit_step(lease, StepCommit(step_id='tool.' + str(lease.token),
                    expected_version=task.version, status='failed',
                    error={'type': 'InvalidToolCall', 'message': 'the admitted task does not carry a valid call'}))
                return
            result = await self._run_leased(call, lease, task)
            if result is not None:
                await self._commit(lease, task, event, result)

    async def _run_leased(self, call: ToolCall, lease: Lease, task: TaskRecord) -> ToolResult | None:
        """Run the call while the lease is renewed; None if the lease was lost."""
        renewer = asyncio.create_task(self._renew(lease))
        running = asyncio.create_task(self.executor.execute(call, lease=lease, task=task))
        try:
            done, _ = await asyncio.wait([running, renewer], return_when=asyncio.FIRST_COMPLETED)
            if running in done:
                return running.result()
            _logger.warning('tool lease lost; the next owner settles the call', exc_info=renewer.exception(),
                            extra={'task_id': lease.task_id, 'tool_call_id': call.tool_call_id})
            return None
        except LeaseLost:
            return None  # the next owner settles the call from the tool journal
        finally:
            for job in (renewer, running):
                job.cancel()
            await asyncio.gather(renewer, running, return_exceptions=True)

    async def _renew(self, lease: Lease) -> None:
        """Keep the lease; a transient store error is retried while the lease still holds."""
        from .engine import LEASE_RENEW_SECONDS
        while True:
            await asyncio.sleep(LEASE_RENEW_SECONDS)
            try:
                lease = await self.store.renew(lease)
            except LeaseLost:
                raise
            except Exception as error:
                if self.clock.now() >= lease.expires_at:
                    raise
                _logger.warning('tool lease renewal failed; retrying', exc_info=error,
                                extra={'task_id': lease.task_id})

    async def _commit(self, lease: Lease, task: TaskRecord, event: Envelope, result: ToolResult) -> None:
        status = _TASK_STATUS[result.status]
        outgoing = [] if status == 'outcome_unknown' else [self._reply(task, event, status, result)]
        try:
            await self.store.commit_step(lease, StepCommit(step_id='tool.' + str(lease.token),
                expected_version=task.version, status=status, result=result.model_dump(mode='json'),
                error=None if status == 'succeeded' else result.error, outgoing=outgoing))
        except LeaseLost:
            pass  # e.g. past the deadline; the tool journal stays authoritative for every caller

    def _reply(self, task: TaskRecord, event: Envelope, status: str, result: ToolResult) -> Envelope:
        """Notify the caller; it reads the output from the journal, so only the status travels."""
        payload = TaskResult(task_id=task.task_id, output=None, responding_agent=task.agent_id, status=status,
                             error=public_error(result.error))
        return Envelope(event_id=str(uuid4()), network_id=event.network_id, kind='tool_result',
            sender=task.agent_id, recipient=event.sender, topic=None, task_id=task.task_id,
            parent_task_id=event.parent_task_id, correlation_id=event.event_id, session_id=event.session_id,
            created_at=self.clock.now(), deadline=event.deadline, traceparent=event.traceparent,
            payload=payload.model_dump(mode='json'))


__all__ = ['ToolWorker', 'default_namespace', 'published_tools']
