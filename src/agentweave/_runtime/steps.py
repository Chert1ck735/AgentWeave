"""The managed AI loop: model turns and tool calls, recoverable by journal replay.

Every sent model request is an attempt in model_attempts and every tool call
an entry in tool_executions, keyed deterministically by task, step and the
provider's call ID. Running the loop again after a crash therefore replays
completed turns and tool results from the journal and resumes at the first
step without a completed turn: the model is not asked twice for the same step
and a completed tool never runs again.
"""
import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid5
from .._records import Clock, Lease, ModelAttempt, Participant, TaskRecord
from ..definitions.agent import Agent
from ..definitions.selections import ToolSelection
from ..definitions.tool import ToolSchema
from ..errors import HandoffRefused, ModelBudgetExceeded, ModelCapacity, ModelError, StoreConflict
from ..messages import Envelope, JSONValue, Message, ToolCall, error_payload
from ..models import ModelRequest, ModelTurn, TextDelta, TurnComplete
from ..ports import Directory, EventBus, ExecutionStore
from ..providers import ModelAdapter, pool_name
from .coordination import (CONSULT_TOOL, ancestry, child_result, consult_envelope, consult_tool, inherited_budget,
                           live_peers, refusal, request_input)
from .conversations import HANDOFF_TOOL, hand_off, handoff_context, handoff_tool, transferred_budget
from .tools import Binding, ToolExecutor, binding_problem, schema_error

ATTEMPT_LEASE_SECONDS = 60
STEPS_PER_OWNER = 1000
_ENDED = frozenset(('succeeded', 'failed', 'cancelled', 'expired', 'outcome_unknown'))
CAPACITY_BACKOFF_SECONDS = (0.05, 1.0)
RETRY_BACKOFF_SECONDS = (1.0, 10.0)
PROGRESS_FLUSH_CHARS = 64
_logger = logging.getLogger('agentweave.runtime')

Progress = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True)
class LoopOutcome:
    """How a managed loop ended; messages are this run's transcript for session history."""
    status: str
    output: JSONValue = None
    error: JSONValue = None
    messages: tuple[Message, ...] = ()


def _model_failure(error: Exception) -> dict:
    payload = error_payload(error)
    if isinstance(error, ModelError):
        payload['kind'] = error.kind
    return payload


def _alias(name: str) -> str:
    """Provider function names allow letters, digits, '_' and '-' only."""
    return name.replace('.', '_')


class ModelLoop:
    def __init__(self, agent: Agent, store: ExecutionStore, adapter: ModelAdapter, *, instance_id: str,
                 clock: Clock, tools: ToolExecutor | None = None, max_model_calls: int = 20,
                 pool_limit: int = 4, history: list[Message] | tuple = (), progress: Progress | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 directory: Directory | None = None, bus: EventBus | None = None, coordinator=None) -> None:
        self.agent, self.store, self.adapter = agent, store, adapter
        self.instance_id, self.clock, self.tools = instance_id, clock, tools
        self.max_model_calls, self.pool_limit = max_model_calls, pool_limit
        self.history, self.progress, self._sleep = list(history), progress, sleep
        self.directory, self.bus = directory, bus
        self.bindings: dict[str, Binding] = (
            {_alias(binding.name): binding for binding in agent.tools} if isinstance(agent.tools, list) else {})
        self.discover = isinstance(agent.tools, ToolSelection)
        # Consulting peers needs a peer scope and a runtime that can wait for answers.
        self.coordinator = coordinator if agent.peers is not None else None
        self.budget = agent.budget
        self.pool = pool_name(agent.model)

    def close(self) -> None:
        if self.tools is not None:
            self.tools.close()

    async def run(self, task: TaskRecord, lease: Lease) -> LoopOutcome:
        event = await self.store.get_input_envelope(task.task_id)
        base = ([Message(role='system', content=self.agent.instructions)] if self.agent.instructions else [])
        base += self.history
        # A consult inherits the root's budget; its model calls count against the same root.
        self.budget = inherited_budget(event) or self._transferred(task) or self.agent.budget
        transcript = [Message(role='user', content=request_input(event, task.input))]
        local = [binding.schema.model_copy(update={'tool_id': alias}) for alias, binding in self.bindings.items()]
        # A new owner after a handoff starts its own steps: the journal keys never clash.
        step = task.handoffs * STEPS_PER_OWNER
        while True:
            catalog, peers = await self._catalog(lease, step)
            schemas = local + [ToolSchema(tool_id=alias, revision=entry.revision, description=entry.description,
                                          input_schema=entry.input_schema, output_schema=entry.output_schema)
                               for alias, entry in catalog.items()]
            if peers:
                schemas += [consult_tool(peers), handoff_tool(peers)]
            request = ModelRequest(messages=base + transcript, tools=schemas,
                                   model_id=self.agent.model.model_id, deadline=task.deadline)
            try:
                self.adapter.validate_request(request)
            except ModelError as error:
                return LoopOutcome('failed', error=_model_failure(error), messages=tuple(transcript))
            turn = await self._turn(task, lease, step, request)
            if isinstance(turn, LoopOutcome):
                return LoopOutcome(turn.status, turn.output, turn.error, tuple(transcript))
            attempt_id, turn = turn
            transcript += turn.messages
            if not turn.tool_calls:
                final = next((m.content for m in reversed(turn.messages) if m.role == 'assistant'), None)
                if final is None or final == '':
                    return LoopOutcome('failed', error={'type': 'EmptyModelResponse',
                        'message': 'the model ended without an answer or a tool call'}, messages=tuple(transcript))
                return LoopOutcome('succeeded', output=final, messages=tuple(transcript))
            if not self._fresh(event):
                # The answer is journaled, but its tools may not start on stale input.
                return LoopOutcome('expired', messages=tuple(transcript))
            if any(call.name == HANDOFF_TOOL for call in turn.tool_calls) and peers:
                handed = await self._handoff(task, lease, event, turn, peers, transcript)
                if handed is None:
                    return LoopOutcome('handed_off', messages=tuple(transcript))  # the next owner answers
                transcript.append(Message(role='tool', tool_results=handed))
                step += 1
                continue
            results = await self._run_tools(task, lease, event, attempt_id, turn, catalog, peers)
            if results is None:
                return LoopOutcome('waiting_peer', messages=tuple(transcript))  # resumed once every answer is in
            transcript.append(Message(role='tool', tool_results=results))
            unknown = next((r for r in results if r['status'] == 'outcome_unknown'), None)
            if unknown is not None:
                return LoopOutcome('outcome_unknown', error=unknown['error'], messages=tuple(transcript))
            step += 1

    async def _turn(self, task: TaskRecord, lease: Lease, step: int,
                    request: ModelRequest) -> tuple[str, ModelTurn] | LoopOutcome:
        """Replay a completed step or send it, retrying transient failures within the step."""
        request_hash = hashlib.sha256(request.model_dump_json(exclude={'deadline'}).encode()).hexdigest()
        retry_delay = RETRY_BACKOFF_SECONDS[0]
        while True:
            if self._remaining(task) <= 0:
                return _past_deadline()
            attempt = await self._reserve(task, lease, step, request_hash)
            if isinstance(attempt, LoopOutcome):
                return attempt
            if attempt.status == 'completed':
                return attempt.attempt_id, ModelTurn.model_validate(attempt.turn)
            try:
                # The task deadline bounds the stream itself, not only the start of it.
                async with asyncio.timeout(self._remaining(task)):
                    turn = await self._generate(attempt, request)
            except TimeoutError:
                await self.store.fail_model_attempt(attempt.attempt_id, {'type': 'ModelError', 'kind': 'timeout',
                    'message': 'the model request outlived the task deadline'}, usage_unknown=True)
                return _past_deadline()
            except ModelError as error:
                await self._fail(attempt, error)
                if not error.retryable:
                    return LoopOutcome('failed', error=_model_failure(error))
                await self._sleep(min(retry_delay, max(0.0, self._remaining(task))))
                retry_delay = min(retry_delay * 2, RETRY_BACKOFF_SECONDS[1])
                continue
            input_tokens, output_tokens = turn.usage.get('input_tokens'), turn.usage.get('output_tokens')
            cost, revision = self._cost(input_tokens, output_tokens)
            finished = await self.store.finish_model_attempt(attempt.attempt_id, turn, input_tokens=input_tokens,
                output_tokens=output_tokens, cost_usd=cost, pricing_revision=revision)
            if finished.status != 'completed':
                # Another owner closed this attempt: its turn is not journaled, so never act on it.
                return LoopOutcome('failed', error={'type': 'AttemptSuperseded',
                                                    'message': 'the model attempt was closed by another owner'})
            return attempt.attempt_id, turn

    def _cost(self, input_tokens, output_tokens) -> tuple[str | None, str | None]:
        pricing = self.budget.pricing if self.budget else None
        if pricing is None:
            return None, None
        cost = pricing.cost(input_tokens, output_tokens)
        return (None if cost is None else str(cost)), pricing.revision

    async def _fail(self, attempt: ModelAttempt, error: ModelError) -> None:
        usage = error.usage or {}
        input_tokens, output_tokens = usage.get('input_tokens'), usage.get('output_tokens')
        cost, _ = self._cost(input_tokens, output_tokens) if usage else (None, None)
        await self.store.fail_model_attempt(attempt.attempt_id, _model_failure(error),
            usage_unknown=error.usage_unknown, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost)

    def _remaining(self, task: TaskRecord) -> float:
        return (task.deadline - self.clock.now()).total_seconds()

    async def _reserve(self, task: TaskRecord, lease: Lease, step: int,
                       request_hash: str) -> ModelAttempt | LoopOutcome:
        delay = CAPACITY_BACKOFF_SECONDS[0]
        while True:
            try:
                return await self.store.reserve_model_attempt(lease, step=step, request_hash=request_hash,
                    pool_name=self.pool, pool_limit=self.pool_limit, budget=self.budget,
                    max_calls=self.max_model_calls, instance_id=self.instance_id,
                    ttl_seconds=ATTEMPT_LEASE_SECONDS)
            except ModelCapacity:
                if self.clock.now() >= task.deadline:
                    return LoopOutcome('expired', error={'type': 'ModelCapacity',
                                                         'message': 'the provider pool stayed full until the deadline'})
                await self._sleep(delay)
                delay = min(delay * 2, CAPACITY_BACKOFF_SECONDS[1])
            except (ModelBudgetExceeded, ModelError, StoreConflict) as error:
                return LoopOutcome('failed', error=_model_failure(error))

    async def _generate(self, attempt: ModelAttempt, request: ModelRequest) -> ModelTurn:
        """Stream one request; deltas are progress only, the turn is the final event."""
        renewer = asyncio.create_task(self._renew(attempt.attempt_id))
        buffer, turn = [], None
        try:
            async for event in self.adapter.generate(request):
                if isinstance(event, TextDelta):
                    buffer.append(event.text)
                    if sum(map(len, buffer)) >= PROGRESS_FLUSH_CHARS:
                        await self._flush(attempt.attempt_id, buffer)
                elif isinstance(event, TurnComplete):
                    turn = event.turn
            await self._flush(attempt.attempt_id, buffer)
        except ModelError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            # An adapter bug after sending: the request may have been billed.
            raise ModelError('unavailable', f'adapter failed: {type(error).__name__}',
                             retryable=False, usage_unknown=True) from error
        finally:
            renewer.cancel()
            await asyncio.gather(renewer, return_exceptions=True)
        if turn is None:
            raise ModelError('unavailable', 'the model stream ended without a complete turn',
                             retryable=True, usage_unknown=True)
        return turn

    async def _flush(self, attempt_id: str, buffer: list[str]) -> None:
        if buffer and self.progress is not None:
            await self.progress(attempt_id, ''.join(buffer))
        buffer.clear()

    async def _renew(self, attempt_id: str) -> None:
        while True:
            await asyncio.sleep(ATTEMPT_LEASE_SECONDS / 3)
            try:
                await self.store.renew_model_attempt(attempt_id, ATTEMPT_LEASE_SECONDS)
            except Exception as error:
                # The pool slot may now expire early; the request itself is unaffected.
                _logger.warning('model attempt renewal failed', exc_info=error, extra={'attempt_id': attempt_id})
                return

    async def _catalog(self, lease: Lease, step: int) -> tuple[dict[str, Participant], list[str]]:
        """The discovered tools and consultable peers of one step, frozen so a replay offers the same.

        Only tools this agent may call and whose executor admits it as a caller
        are offered; the executor checks the caller again before any effect.
        Only peers of this agent's scope that answer requests now may be consulted.
        """
        if not self.discover and self.coordinator is None:
            return {}, []
        snapshot = self.directory.snapshot() if self.directory else {}
        offered = []
        for logical_id, instances in sorted(snapshot.items()):
            entry = instances[0]
            if (self.discover and entry.kind == 'tool' and self.tools is not None
                    and self.tools.policy.can_tool(self.agent.name, logical_id)
                    and (entry.allowed_callers is None or self.agent.name in entry.allowed_callers)):
                offered.append(entry.model_dump(mode='json'))
        if self.coordinator is not None:
            peers = live_peers(self.directory, self.coordinator.worker.policy, self.agent.name)
            offered += [snapshot[name][0].model_dump(mode='json') for name in peers]
        frozen = [Participant.model_validate(entry)
                  for entry in await self.store.freeze_tool_catalog(lease, step=step, catalog=offered)]
        return ({entry.alias: entry for entry in frozen if entry.kind == 'tool'},
                [entry.logical_id for entry in frozen if entry.kind == 'agent'])

    async def _run_tools(self, task: TaskRecord, lease: Lease, event: Envelope, attempt_id: str,
                         turn: ModelTurn, catalog: dict[str, Participant],
                         peers: list[str]) -> list[dict[str, Any]] | None:
        """Validate every proposed call first; dispatch only when all of them are valid.

        Consults of one turn are all sent before waiting, so their peers work
        concurrently; None means the loop now waits durably for their answers.
        """
        ids = [call.provider_call_id for call in turn.tool_calls]
        if len(set(ids)) != len(ids):
            # Each call's journal identity derives from its ID; duplicates are ambiguous.
            return [{'id': call.provider_call_id, 'status': 'invalid', 'output': None,
                     'error': {'type': 'DuplicateCallId', 'message': 'calls in one turn need distinct IDs'}}
                    for call in turn.tool_calls]
        problems = [self._problem(call, catalog, peers) for call in turn.tool_calls]
        if any(problems):
            return [problem or {'id': call.provider_call_id, 'status': 'skipped', 'output': None,
                                'error': {'type': 'Skipped', 'message': 'another call in this turn was invalid'}}
                    for call, problem in zip(turn.tool_calls, problems)]
        results, waiting, requests = [], [], []
        for proposed in turn.tool_calls:
            if peers and proposed.name == CONSULT_TOOL:
                call_id = str(uuid5(UUID(attempt_id), proposed.provider_call_id))
                consulted = await self._consult(task, event, call_id, proposed)
                if isinstance(consulted, dict):
                    results.append(consulted)
                else:
                    waiting.append(consulted[0])
                    requests += [consulted[1]] if consulted[1] is not None else []
                continue
            entry = catalog.get(proposed.name)
            binding = self.bindings.get(proposed.name)
            target, revision = (entry.logical_id, entry.revision) if entry else (binding.name, binding.schema.revision)
            call = ToolCall(tool_call_id=str(uuid5(UUID(attempt_id), proposed.provider_call_id)),
                            tool_id=target, revision=revision,
                            arguments=proposed.arguments, caller=self.agent.name,
                            deadline=task.deadline, expires_at=event.expires_at)
            if entry is not None:
                result = await self.tools.execute_remote(call, entry, bus=self.bus, lease=lease, task=task,
                                                         directory=self.directory)
            else:
                result = await self.tools.execute(call, lease=lease, task=task)
            results.append({'id': proposed.provider_call_id, 'status': result.status,
                            'output': result.output, 'error': result.error})
            if result.status == 'outcome_unknown':
                break  # later calls of this turn must not run after an unknown effect
        if waiting and not any(result['status'] == 'outcome_unknown' for result in results):
            await self.store.begin_consult(lease, requests, children=waiting)
            return None
        return results

    def _transferred(self, task):
        """A handed-off task keeps the root's limits; prices come from this agent's own provider."""
        inherited = transferred_budget(task)
        own = self.agent.budget
        if inherited is not None and own is not None and own.pricing is not None:
            return inherited.model_copy(update={'pricing': own.pricing})
        return inherited

    async def _handoff(self, task, lease, event, turn, peers, transcript) -> list | None:
        """Hand off as the turn's only call; otherwise nothing runs. None once the task moved."""
        def result(call, status, kind, message):
            return {'id': call.provider_call_id, 'status': status, 'output': None,
                    'error': {'type': kind, 'message': message}}
        if len(turn.tool_calls) > 1:
            return [result(call, 'invalid', 'HandoffNotAlone', 'a handoff must be the only call of its turn')
                    if call.name == HANDOFF_TOOL else
                    result(call, 'skipped', 'Skipped', 'the turn also asked for a handoff')
                    for call in turn.tool_calls]
        [call] = turn.tool_calls
        problem = schema_error(handoff_tool(peers).input_schema, call.arguments)
        if problem is not None:
            return [result(call, 'invalid', 'ValidationError', problem)]
        context = handoff_context(self.history + transcript, call.arguments['message'], self.agent.name, self.budget)
        try:
            await hand_off(self.coordinator.worker, lease, task, event, call.arguments['agent'], context, None)
        except HandoffRefused as error:
            return [result(call, 'invalid', 'HandoffRefused', str(error))]
        return None

    async def _consult(self, task: TaskRecord, event: Envelope, call_id: str, proposed):
        """A consult call's answer, or (child_id, request to send or None) while it is pending."""
        child_id = str(uuid5(UUID(call_id), 'consult'))
        child = await self.store.get_task(child_id)
        if child is not None and child.status in _ENDED:
            answer = child_result(child)
            return {'id': proposed.provider_call_id, 'status': answer.status, 'output': answer.output,
                    'error': answer.error}
        if child is not None or await self.store.get_submission_envelope(child_id) is not None:
            return child_id, None  # sent before a restart; still waiting for it
        target = proposed.arguments['agent']
        refused = refusal(self.directory, self.coordinator.worker.policy, self.agent.name, target, ancestry(event))
        if refused is not None:
            return {'id': proposed.provider_call_id, 'status': 'invalid', 'output': None,
                    'error': {'type': refused[0], 'message': refused[1]}}
        return child_id, consult_envelope(event, task, self.agent.name, target, proposed.arguments['input'],
                                          child_id, self.budget, self.clock.now())

    def _problem(self, proposed, catalog: dict[str, Participant], peers: list[str]) -> dict | None:
        def invalid(kind, message):
            return {'id': proposed.provider_call_id, 'status': 'invalid', 'output': None,
                    'error': {'type': kind, 'message': message}}
        if proposed.arguments_error:
            return invalid('InvalidArguments', proposed.arguments_error)
        if peers and proposed.name == CONSULT_TOOL:
            problem = schema_error(consult_tool(peers).input_schema, proposed.arguments)
            return None if problem is None else invalid('ValidationError', problem)
        entry = catalog.get(proposed.name)
        if entry is not None and self.tools is not None:
            problem = schema_error(entry.input_schema, proposed.arguments)
            return None if problem is None else invalid('ValidationError', problem)
        binding = self.bindings.get(proposed.name)
        if binding is None or self.tools is None:
            return invalid('UnknownTool', 'no such tool is available to this agent')
        problem = binding_problem(binding, proposed.arguments)
        return None if problem is None else invalid('ValidationError', problem)

    def _fresh(self, event: Envelope) -> bool:
        now = self.clock.now()
        return now < event.deadline and (event.expires_at is None or now < event.expires_at)


def _past_deadline() -> LoopOutcome:
    return LoopOutcome('expired', error={'type': 'DeadlineExceeded',
                                         'message': 'the task deadline passed during the model step'})


__all__ = ['LoopOutcome', 'ModelLoop']
