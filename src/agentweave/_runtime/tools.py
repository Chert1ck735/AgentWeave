"""One execution path for every tool: validation, hooks, journal, retries, freshness.

A call is validated and approved by before hooks without any journal write.
reserve_tool then records the attempt under the task lease before the effect
starts; finish_tool commits its outcome under the attempt token. A crash in
between leaves the row running, which a later executor turns into a retry for
retry-safe tools and into outcome_unknown otherwise. A settled call is only
replayed from the journal, never executed again.

One executor dispatches a given task lease; calls with the same ID are
serialized inside it. A timed-out sync tool keeps its thread until it returns
(Python cannot stop it), so a retry of a retry-safe tool may overlap it.

A tool published by another process is called through the same journal:
RemoteToolBinding submits the call to the tool's inbox, and the ToolWorker of
that process admits it as its own task and runs it with a ToolExecutor under
the same tool_call_id. Every repetition of one call waits for that outcome.
"""
import asyncio
import inspect
import json
import logging
import random as random_module
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid5
from pydantic import TypeAdapter, ValidationError
from .._policy import Policy
from .._records import Clock, Lease, Participant, SystemClock, TaskRecord, ToolLease
from ..context import ToolContext
from ..definitions.agent import Agent
from ..definitions.tool import RetryPolicy, ToolDefinition, ToolSignature, tool_signature
from ..definitions.toolbox import HttpToolSpec
from ..errors import (BrokerUnavailable, DefinitionError, LeaseLost, StoreConflict, ToolArgumentsInvalid,
                      ToolCallFailed, ToolNotStarted, ToolRequestRejected, ToolRevisionChanged, ToolUnavailable)
from ..hooks import Deny, call_hook
from ..messages import MAX_ENVELOPE_BYTES, Envelope, JSONValue, ToolCall, ToolResult, error_payload
from ..ports import Directory, EventBus, ExecutionStore
from .handlers import HandlerExecutor, HandlerNotStarted

_logger = logging.getLogger('agentweave.tools')
_json_value = TypeAdapter(JSONValue)
_SETTLED = frozenset(('succeeded', 'failed', 'expired', 'outcome_unknown'))
_TERMINAL_TASK = frozenset(('succeeded', 'failed', 'cancelled', 'expired', 'outcome_unknown'))
# A before hook may change arguments only; identity, target and time bounds are fixed.
_PROTECTED_FIELDS = ('tool_call_id', 'tool_id', 'revision', 'caller', 'deadline', 'expires_at')
MIN_INVOCATION_SECONDS = 0.001
REVISION_CHANGED = 'ToolRevisionChanged'
REMOTE_POLL_SECONDS = 0.05
# A Python tool, an HTTP tool, or a runtime tool with invoke() and JSON Schemas (MCP).
Binding = ToolDefinition | HttpToolSpec | Any
# An attempt running at the deadline gets this long to report before the caller settles it.
REMOTE_SETTLE_GRACE_SECONDS = 2


def _output_schema(binding: Binding) -> dict | None:
    """The result schema journaled with a call, so an operator resolution is checked against it."""
    schema = getattr(binding, 'schema', None)
    return schema.output_schema if schema is not None else getattr(binding, 'output_schema', None)


def _rejection(call: ToolCall, status: str, kind: str, message: str) -> ToolResult:
    return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id, status=status,
                      error={'type': kind, 'message': message})


def _json_output(value: Any) -> JSONValue:
    value = _json_value.validate_python(value)
    if len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()) > MAX_ENVELOPE_BYTES:
        raise ValueError('tool result exceeds 1 MiB')
    return value


class ToolExecutor:
    def __init__(self, agent: Agent, store: ExecutionStore, *, instance_id: str, clock: Clock,
                 policy: Policy | None = None, owner: tuple | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 random: Callable[[], float] = random_module.random) -> None:
        self.agent, self.store, self.instance_id, self.clock = agent, store, instance_id, clock
        self.policy = policy or Policy([agent])
        self.tools: dict[str, Binding] = (
            {binding.name: binding for binding in agent.tools} if isinstance(agent.tools, list) else {})
        self._sleep, self._random = sleep, random
        self._calls: dict[str, list] = {}  # tool_call_id -> [lock, waiters]
        # A stable owner keeps threads of timed-out sync tools counted across Worker replacements.
        self.executor = HandlerExecutor(agent.concurrency, owner=owner)

    def close(self) -> None:
        self.executor.close()

    async def execute(self, call: ToolCall, *, lease: Lease, task: TaskRecord) -> ToolResult:
        """Run call once, or return its journaled outcome if it already settled."""
        entry = self._calls.setdefault(call.tool_call_id, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                return await self._execute(call, lease, task)
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._calls.pop(call.tool_call_id, None)

    async def _execute(self, call: ToolCall, lease: Lease, task: TaskRecord) -> ToolResult:
        saved = await self.store.get_tool_execution(call.tool_call_id)
        if saved is not None and saved.requested != call.model_dump(mode='json'):
            # One ID, two different calls: never answer one with the other's result.
            return _rejection(call, 'invalid', 'ConflictingCallId',
                              'this tool_call_id already identifies a different call')
        if saved is not None and saved.status in _SETTLED:
            return await self._settled(call, saved, task)
        binding = self.tools.get(call.tool_id)
        if saved is not None:
            # Resume only if the tool, permission and approved arguments still hold.
            rejected = self._check(call, binding, saved.arguments)
            if rejected is not None:
                return rejected
            return await self._dispatch(call, binding, saved.arguments, lease, task)
        if not self._fresh(call):
            return _rejection(call, 'expired', 'InputExpired', 'input expired before the tool started')
        prepared = await self._prepare(call, binding, task)
        if isinstance(prepared, ToolResult):
            return prepared
        return await self._dispatch(call, binding, prepared, lease, task)

    async def _prepare(self, call: ToolCall, binding: ToolDefinition | None,
                       task: TaskRecord) -> dict | ToolResult:
        """Check target, permission and arguments, then apply before hooks; no journal write."""
        rejected = self._check(call, binding, call.arguments)
        if rejected is not None:
            return rejected
        return await self._approve(call, task, lambda arguments: self._validate_binding(call, binding, arguments))

    async def _approve(self, call: ToolCall, task: TaskRecord,
                       validate: Callable[[dict], ToolResult | None]) -> dict | ToolResult:
        """Run before hooks; changed arguments are validated again before they may run."""
        approved = call
        context = self._context(call, task, attempt=0)
        for hook in self.agent.before_tool_hooks:
            try:
                outcome = await call_hook(hook, approved, context)
            except Exception as error:
                # A failing check fails closed: no effect without an approval.
                _logger.error('before_tool hook failed', exc_info=error, extra={'tool_call_id': call.tool_call_id})
                return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id,
                                  status='denied', error=error_payload(error))
            if isinstance(outcome, Deny):
                return _rejection(call, 'denied', 'Denied', outcome.reason)
            if not isinstance(outcome, ToolCall):
                return _rejection(call, 'denied', 'HookError', 'before_tool must return a ToolCall or Deny')
            if any(getattr(outcome, field) != getattr(call, field) for field in _PROTECTED_FIELDS):
                return _rejection(call, 'denied', 'ProtectedField',
                                  'before_tool may change arguments only')
            invalid = validate(outcome.arguments)
            if invalid is not None:
                return invalid
            approved = outcome
        return dict(approved.arguments)

    def _check(self, call: ToolCall, binding: ToolDefinition | None, arguments: dict) -> ToolResult | None:
        if binding is None:
            return _rejection(call, 'invalid', 'UnknownTool', 'tool is not available to this agent')
        if call.revision != binding.schema.revision:
            return _rejection(call, 'invalid', REVISION_CHANGED, 'tool schema changed; refresh and call again')
        if not self._may_call(call, binding):
            return _rejection(call, 'denied', 'NotAllowed', 'caller may not use this tool')
        return self._validate_binding(call, binding, arguments)

    def _validate_binding(self, call: ToolCall, binding: Binding, arguments: dict) -> ToolResult | None:
        """Python tools validate against their typed signature, the others against their JSON Schema."""
        if not isinstance(binding, ToolDefinition):
            problem = schema_error(binding.input_schema, arguments)
            return None if problem is None else _rejection(call, 'invalid', 'ValidationError', problem)
        return self._validate(call, tool_signature(binding.callable), arguments)

    def _may_call(self, call: ToolCall, binding: ToolDefinition) -> bool:
        return call.caller == self.agent.name and self.policy.can_tool(call.caller, call.tool_id)

    async def execute_remote(self, call: ToolCall, participant: Participant, *, bus: EventBus | None,
                             lease: Lease, task: TaskRecord, directory: Directory | None = None) -> ToolResult:
        """Call a discovered tool of another process after this agent's own checks.

        Permission, the executor's published allowed_callers, the argument
        schema and before hooks are checked here; the executor checks the
        caller, revision and arguments again before any effect. A replay finds
        the call already sent and waits for its saved outcome instead. After
        hooks run once on a succeeded result and are journaled with it.
        """
        remote = RemoteToolBinding(participant, bus, self.store, directory=directory, clock=self.clock)
        sent = await remote.sent(call)
        if isinstance(sent, ToolResult):
            return sent
        if sent is None:
            rejected = self._check_remote(call, participant)
            if rejected is not None:
                return rejected
            approved = await self._approve(call, task, lambda arguments: _schema_problem(call, participant, arguments))
            if isinstance(approved, ToolResult):
                return approved
            sent = call.model_copy(update={'arguments': approved})
        result = await remote.execute(sent, parent_task_id=task.task_id, session_id=task.session_id)
        if result.status != 'succeeded' or not self.agent.after_tool_hooks:
            return result
        row = await self.store.get_tool_execution(call.tool_call_id)
        return await self._settled(call, row, task) if row is not None else result

    def _check_remote(self, call: ToolCall, participant: Participant) -> ToolResult | None:
        if call.tool_id != participant.logical_id or call.revision != participant.revision:
            return _rejection(call, 'invalid', 'UnknownTool', 'the call does not match the chosen tool')
        callers = participant.allowed_callers
        if (call.caller != self.agent.name or not self.policy.can_tool(call.caller, call.tool_id)
                or (callers is not None and call.caller not in callers)):
            return _rejection(call, 'denied', 'NotAllowed', 'caller may not use this tool')
        return _schema_problem(call, participant, call.arguments)

    @staticmethod
    def _validate(call: ToolCall, signature: ToolSignature, arguments: dict) -> ToolResult | None:
        try:
            signature.parse(arguments)
        except ValidationError as error:
            return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id,
                              status='invalid', error=error_payload(error))
        return None

    async def _dispatch(self, call: ToolCall, binding: ToolDefinition, arguments: dict,
                        lease: Lease, task: TaskRecord) -> ToolResult:
        retry = binding.retry
        while True:
            if not self._fresh(call):
                settled = await self.store.expire_tool(call.tool_call_id, lease=lease)
                if settled is None:
                    return _rejection(call, 'expired', 'InputExpired', 'input expired before the tool started')
                return await self._settled(call, settled, task)
            attempt = await self.store.reserve_tool(call, arguments, lease=lease, instance_id=self.instance_id,
                                                    retry_safe=retry.safe, max_attempts=retry.max_attempts,
                                                    output_schema=_output_schema(binding))
            if attempt.status != 'running':
                return await self._settled(call, attempt, task)
            context = self._context(call, task, attempt=attempt.attempt)
            # expires_at gates the start only; a call sent while fresh may finish after it.
            timeout = min(self._attempt_limit(binding), max(MIN_INVOCATION_SECONDS, self._until_deadline(call)))
            try:
                raw = await self.executor.invoke(self._invocation(binding, call), context, arguments, timeout)
            except HandlerNotStarted:
                finished = await self.store.finish_tool(attempt, status='expired')
                return await self._settled(call, finished, task)
            except (LeaseLost, asyncio.CancelledError):
                raise
            except Exception as error:
                settled = await self._failed_attempt(call, binding, attempt, error)
                if settled is None:
                    continue
                return await self._settled(call, settled, task)
            try:
                output = self._output(binding, raw)
            except (ValueError, TypeError):
                # The effect ran but its result is unusable: unknown, never a clean failure.
                finished = await self.store.mark_tool_unknown(attempt, error={
                    'type': 'InvalidToolOutput', 'message': 'tool result does not match its declared output type'})
                return await self._settled(call, finished, task)
            finished = await self.store.finish_tool(attempt, status='succeeded', result=output)
            return await self._settled(call, finished, task)

    async def _failed_attempt(self, call: ToolCall, binding: ToolDefinition, attempt: ToolLease,
                              error: Exception) -> ToolLease | None:
        """Settle a raised attempt, or return None after scheduling a retry-safe retry."""
        failure = error_payload(error)
        _logger.error('tool attempt failed', exc_info=error,
                      extra={'tool_call_id': call.tool_call_id, 'tool_id': call.tool_id, 'attempt': attempt.attempt})
        if isinstance(error, (ToolRequestRejected, ToolArgumentsInvalid, ToolRevisionChanged)):
            # The tool answered with a refusal, or nothing was sent: known, and never repeated.
            return await self.store.finish_tool(attempt, status='failed', error=failure)
        if isinstance(error, ToolNotStarted) and not attempt.retry_safe:
            return await self.store.finish_tool(attempt, status='failed', error=failure)
        # The journaled policy decides, so a redeploy cannot change a call in flight.
        if not attempt.retry_safe:
            # The tool may have acted before raising; never repeat it blindly.
            return await self.store.mark_tool_unknown(attempt, error=failure)
        delay = self._backoff(binding.retry, attempt.attempt)
        if attempt.attempt < attempt.max_attempts and self._fresh(call, after=delay):
            await self.store.release_tool(attempt, error=failure)
            await self._sleep(delay)
            return None
        return await self.store.finish_tool(attempt, status='failed', error=failure)

    async def _settled(self, call: ToolCall, row: ToolLease, task: TaskRecord) -> ToolResult:
        if row.status != 'succeeded':
            return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id,
                              status=row.status, error=row.error)
        if not self.agent.after_tool_hooks:
            return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id,
                              status='succeeded', output=row.result)
        if row.hook_status is None:
            row = await self._after(call, row, task)
        if row.hook_status == 'failed':
            return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id,
                              status='failed', error=row.hook_error)
        return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id,
                          status='succeeded', output=row.hook_result)

    async def _after(self, call: ToolCall, row: ToolLease, task: TaskRecord) -> ToolLease:
        """Shape a committed result; a hook failure is journaled, the effect is not repeated.

        Hooks see the call with the arguments that actually ran; the arguments
        first requested stay available as context.requested_arguments.
        """
        executed = call.model_copy(update={'arguments': row.arguments})
        output = row.result
        context = self._context(executed, task, attempt=row.attempt, requested=call.arguments)
        try:
            for hook in self.agent.after_tool_hooks:
                output = _json_output(await call_hook(hook, executed, output, context))
        except Exception as error:
            _logger.error('after_tool hook failed', exc_info=error, extra={'tool_call_id': call.tool_call_id})
            return await self.store.record_tool_hook(call.tool_call_id, status='failed', error=error_payload(error))
        return await self.store.record_tool_hook(call.tool_call_id, status='succeeded', result=output)

    @staticmethod
    def _attempt_limit(binding: Binding) -> float:
        """An HTTP or MCP tool's timeout bounds its whole attempt; a Python tool's retry policy does."""
        return binding.retry.timeout if isinstance(binding, ToolDefinition) else binding.timeout

    def _invocation(self, binding: Binding, call: ToolCall) -> Callable:
        """The (context, arguments) callable that performs one attempt's effect."""
        if isinstance(binding, HttpToolSpec):
            from ..adapters.tools.http import HttpBinding
            http = HttpBinding(binding)

            async def invoke_http(context, arguments):
                return await http.invoke(call.model_copy(update={'arguments': arguments}), context)
            return invoke_http
        if isinstance(binding, ToolDefinition):
            return self._bound(binding, tool_signature(binding.callable))

        async def invoke_bound(context, arguments):  # e.g. an MCP tool holding its own connection
            return await binding.invoke(call.model_copy(update={'arguments': arguments}), context)
        return invoke_bound

    @staticmethod
    def _output(binding: Binding, raw: Any) -> JSONValue:
        """The journaled result; raises ValueError or TypeError when it breaks the declared output."""
        if not isinstance(binding, ToolDefinition):
            output = _json_output(raw)
            problem = schema_error(binding.output_schema, output)
            if problem is not None:
                raise ValueError(problem)
            return output
        signature = tool_signature(binding.callable)
        return _json_output(signature.output.dump_python(signature.output.validate_python(raw, strict=True), mode='json'))

    @staticmethod
    def _bound(binding: ToolDefinition, signature: ToolSignature) -> Callable:
        """Adapt a typed tool to the executor's (context, arguments) calling convention."""
        function = binding.callable

        def keywords(context: ToolContext, arguments: dict) -> dict:
            values = signature.keywords(arguments)
            if signature.context_parameter:
                values[signature.context_parameter] = context
            return values

        if inspect.iscoroutinefunction(function):
            async def invoke_async(context, arguments):
                return await function(**keywords(context, arguments))
            return invoke_async

        def invoke(context, arguments):
            return function(**keywords(context, arguments))
        return invoke

    def _context(self, call: ToolCall, task: TaskRecord, *, attempt: int,
                 requested: dict | None = None) -> ToolContext:
        return ToolContext(call, task_id=task.task_id, session_id=task.session_id,
                           attempt=attempt, clock=self.clock, requested_arguments=requested)

    def _until_deadline(self, call: ToolCall) -> float:
        return (call.deadline - self.clock.now()).total_seconds()

    def _remaining(self, call: ToolCall) -> float:
        remaining = self._until_deadline(call)
        if call.expires_at is not None:
            remaining = min(remaining, (call.expires_at - self.clock.now()).total_seconds())
        return remaining

    def _fresh(self, call: ToolCall, *, after: float = 0) -> bool:
        return self._remaining(call) > after

    def _backoff(self, retry: RetryPolicy, attempt: int) -> float:
        """Exponential delay from min_delay, spread up to twice that step, capped at max_delay."""
        delay = min(retry.max_delay, retry.min_delay * 2 ** (attempt - 1))
        if retry.jitter:
            delay += (min(retry.max_delay, delay * 2) - delay) * self._random()
        return delay


def remote_task_id(tool_call_id: str) -> str:
    """The executor's task for a remote call; every repetition of the call maps to it."""
    return str(uuid5(UUID(tool_call_id), 'tool-task'))


def schema_error(schema: dict, arguments: dict) -> str | None:
    """Why arguments break a published JSON Schema, or None when they conform."""
    from jsonschema import Draft202012Validator, SchemaError
    from jsonschema.exceptions import best_match
    try:
        error = best_match(Draft202012Validator(schema).iter_errors(arguments))
    except SchemaError:
        return 'the published tool schema is not a valid JSON Schema'
    return None if error is None else error.message[:1000]


def binding_problem(binding: Binding, arguments: dict) -> str | None:
    """Why arguments do not fit an agent's own tool, or None when they do."""
    if not isinstance(binding, ToolDefinition):
        return schema_error(binding.input_schema, arguments)
    try:
        tool_signature(binding.callable).parse(arguments)
    except ValidationError as error:
        return str(error)
    return None


def _schema_problem(call: ToolCall, participant: Participant, arguments: dict) -> ToolResult | None:
    problem = schema_error(participant.input_schema, arguments)
    return None if problem is None else _rejection(call, 'invalid', 'ValidationError', problem)


def _row_result(call: ToolCall, row: ToolLease) -> ToolResult:
    succeeded = row.status == 'succeeded'
    return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id, status=row.status,
                      output=row.result if succeeded else None, error=None if succeeded else row.error)


def _task_result(call: ToolCall, task: TaskRecord) -> ToolResult:
    """The outcome an executor committed for a call that never reached the tool journal."""
    try:
        result = ToolResult.model_validate(task.result)
    except (ValidationError, TypeError):
        result = None
    if result is not None and (result.tool_call_id, result.tool_id) == (call.tool_call_id, call.tool_id):
        return result
    status = task.status if task.status in ('expired', 'outcome_unknown') else 'failed'
    return ToolResult(tool_call_id=call.tool_call_id, tool_id=call.tool_id, status=status, error=task.error)


class RemoteToolBinding:
    """A discovered tool of another process, called through its inbox and the shared journal.

    schema is the catalog entry the caller chose. The call is a durable
    submission: the outbox delivers it until the executor's receipt, and the
    executor journals the outcome under the same tool_call_id. The caller waits
    for that outcome at most until the call's deadline. Neither admission nor a
    new attempt can start after the deadline, so a read then is final: an
    unadmitted call never ran (ToolUnavailable). An attempt still running gets a
    short grace and is then settled once as outcome_unknown in the journal, so
    a late finish cannot give the same call a second, different answer.
    """

    def __init__(self, schema: Participant, bus: EventBus | None, store: ExecutionStore, *,
                 directory: Directory | None = None, clock: Clock | None = None) -> None:
        if schema.kind != 'tool':
            raise DefinitionError('a remote tool binding needs a tool catalog entry')
        self.schema, self.bus, self.store, self.directory = schema, bus, store, directory
        self.clock = clock or getattr(store, 'clock', None) or SystemClock()

    async def invoke(self, call: ToolCall) -> JSONValue:
        """Return the tool's output, or raise why the call produced none."""
        result = await self.execute(call)
        if result.status == 'succeeded':
            return result.output
        kind = result.error.get('type') if isinstance(result.error, dict) else None
        message = result.error.get('message', '') if isinstance(result.error, dict) else ''
        if kind == REVISION_CHANGED:
            raise ToolRevisionChanged(message)
        if kind == 'ToolUnavailable':
            raise ToolUnavailable(message)
        raise ToolCallFailed(result)

    async def execute(self, call: ToolCall, *, parent_task_id: str | None = None,
                      session_id: str | None = None) -> ToolResult:
        if call.tool_id != self.schema.logical_id:
            raise ValueError('the call targets another tool than this binding')
        sent = await self.sent(call)
        if isinstance(sent, ToolResult):
            return sent
        if sent is None:
            refused = self._refusal(call)
            if refused is not None:
                return refused
            conflict = await self._submit(call, parent_task_id, session_id)
            if conflict is not None:
                return conflict
        return await self._outcome(call)

    async def sent(self, call: ToolCall) -> ToolCall | ToolResult | None:
        """The call as first submitted, None if never sent, or a result for a conflicting ID.

        A submitted call keeps the arguments its before hooks approved then.
        """
        envelope = await self.store.get_submission_envelope(remote_task_id(call.tool_call_id))
        if envelope is None:
            return None
        try:
            submitted = ToolCall.model_validate(envelope.payload)
        except ValidationError:
            submitted = None
        if (submitted is None or envelope.kind != 'tool_call' or
                any(getattr(submitted, field) != getattr(call, field) for field in _PROTECTED_FIELDS)):
            return _rejection(call, 'invalid', 'ConflictingCallId', 'this tool_call_id already identifies a different call')
        return submitted

    def _refusal(self, call: ToolCall) -> ToolResult | None:
        """Refuse before sending what cannot run: stale input, no executor or a changed schema."""
        now = self.clock.now()
        if now >= call.deadline or (call.expires_at is not None and now >= call.expires_at):
            return _rejection(call, 'expired', 'InputExpired', 'input expired before the tool started')
        if call.revision != self.schema.revision:
            return _rejection(call, 'invalid', REVISION_CHANGED, 'tool schema changed; refresh and call again')
        if self.directory is None:
            return None
        live = [p for p in self.directory.snapshot().get(call.tool_id, ()) if p.kind == 'tool']
        if not live:
            return _rejection(call, 'failed', 'ToolUnavailable', 'no live executor publishes this tool')
        if live[0].revision != call.revision:
            return _rejection(call, 'invalid', REVISION_CHANGED, 'tool schema changed; refresh and call again')
        return None

    async def _submit(self, call: ToolCall, parent_task_id: str | None, session_id: str | None) -> ToolResult | None:
        envelope = Envelope(event_id=str(uuid5(UUID(call.tool_call_id), 'tool-call')),
            network_id=self.schema.network_id, kind='tool_call', sender=call.caller, recipient=call.tool_id,
            topic=None, task_id=remote_task_id(call.tool_call_id), parent_task_id=parent_task_id,
            correlation_id=None, session_id=session_id, created_at=self.clock.now(), deadline=call.deadline,
            traceparent=None, payload=call.model_dump(mode='json'), expires_at=call.expires_at)
        try:
            await self.store.submit(envelope)
        except StoreConflict:
            # Another sender of this ID was first; wait for its call only if it is this one.
            sent = await self.sent(call)
            return sent if isinstance(sent, ToolResult) else None
        if self.bus is not None:
            try:
                await self.bus.publish(envelope)  # saves the outbox's first delay; the outbox stays authoritative
            except (BrokerUnavailable, TimeoutError) as error:
                _logger.info('tool call left to the outbox: %s', error, extra={'tool_call_id': call.tool_call_id})
        return None

    async def _outcome(self, call: ToolCall) -> ToolResult:
        task_id = remote_task_id(call.tool_call_id)
        while True:
            now = self.clock.now()
            row = await self.store.get_tool_execution(call.tool_call_id)
            if row is not None and row.status in _SETTLED:
                return _row_result(call, row)
            task = await self.store.get_task(task_id)
            if task is not None and task.status in _TERMINAL_TASK:
                if row is None:
                    return _task_result(call, task)
                # The executor may have settled the row after the read above.
                return await self._abandoned(call)
            if task is None and call.expires_at is not None and now >= call.expires_at:
                # Admission refuses expired input, so this call can never start.
                return _rejection(call, 'expired', 'InputExpired', 'input expired before an executor took the call')
            if now >= call.deadline:
                if task is None:
                    return _rejection(call, 'failed', 'ToolUnavailable', 'no executor took the call before its deadline')
                if row is None:
                    return _rejection(call, 'expired', 'InputExpired', 'the call expired before its tool started')
                if now >= call.deadline + timedelta(seconds=REMOTE_SETTLE_GRACE_SECONDS):
                    return await self._abandoned(call)
            await asyncio.sleep(REMOTE_POLL_SECONDS)

    async def _abandoned(self, call: ToolCall) -> ToolResult:
        """Settle a call nobody finished in time, once; an outcome recorded first wins."""
        row = await self.store.abandon_tool(call.tool_call_id)
        return _row_result(call, row)


__all__ = ['RemoteToolBinding', 'ToolExecutor', 'binding_problem', 'remote_task_id', 'schema_error']
