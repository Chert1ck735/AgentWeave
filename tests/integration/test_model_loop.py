"""Durable managed AI loop: replay, tools, budgets, provider pool, retries, expiry."""
import asyncio
import sqlite3
from datetime import timedelta
from uuid import uuid4
import pytest
from tests.integration.test_python_runtime import eventually, runtime_harness  # noqa: F401

INSTANCE = '00000000-0000-0000-0000-0000000000b1'


class SimulatedCrash(BaseException):
    """Stands in for the worker process dying."""


class ModelHarness:
    def __init__(self, store, clock, envelope_factory):
        self.store, self.clock, self.envelope_factory = store, clock, envelope_factory
        self.tool_effects = []
        self.agent = self.model = self.task = self.lease = self.outcome = None

    async def begin(self, agent, model, payload='12 + 7?', **changes):
        self.agent, self.model = agent, model
        event = self.envelope_factory(recipient=agent.name, payload=payload, **changes)
        await self.store.admit(event, 'def-v1')
        self.lease = await self.store.claim(event.task_id, INSTANCE, 30)
        self.task = await self.store.get_task(event.task_id)
        return self

    async def sleep(self, seconds):
        self.clock.advance(seconds)

    def loop(self, **limits):
        from agentweave._runtime.steps import ModelLoop
        from agentweave._runtime.tools import ToolExecutor
        tools = ToolExecutor(self.agent, self.store, instance_id=INSTANCE, clock=self.clock, sleep=self.sleep)
        return ModelLoop(self.agent, self.store, self.model, instance_id=INSTANCE, clock=self.clock,
                         tools=tools, sleep=self.sleep, **limits)

    async def run(self, **limits):
        loop = self.loop(**limits)
        try:
            self.outcome = await loop.run(self.task, self.lease)
        finally:
            loop.close()
        return self.outcome

    def attempts(self):
        with sqlite3.connect(self.store.path) as connection:
            return connection.execute('SELECT status, input_tokens, output_tokens, usage_known, cost_usd '
                                      'FROM model_attempts WHERE task_id=? ORDER BY step, attempt',
                                      (self.task.task_id,)).fetchall()


@pytest.fixture
def harness(store, fake_clock, envelope_factory):
    return ModelHarness(store, fake_clock, envelope_factory)


def calculator(harness, **agent_options):
    from agentweave import Agent, Model, tool

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        harness.tool_effects.append((a, b))
        return a + b
    return lambda model: Agent('assistant', instructions='Use the calculator.', tools=[add],
                               model=Model.custom(model), **agent_options)


async def test_text_turn_completes_with_instructions_and_input(harness):
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([text_turn('hello')])
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run()
    assert outcome.status == 'succeeded' and outcome.output == 'hello'
    sent = model.requests[0].messages
    assert (sent[0].role, sent[0].content) == ('system', 'Use the calculator.')
    assert (sent[-1].role, sent[-1].content) == ('user', '12 + 7?')
    assert [t.tool_id for t in model.requests[0].tools] == ['add']


async def test_tool_round_trip_returns_the_result_to_the_model(harness):
    from agentweave.testing import FakeModel, text_turn, tool_turn
    model = FakeModel([tool_turn('add', {'a': 12, 'b': 7}), text_turn('19')])
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run()
    assert outcome.status == 'succeeded' and outcome.output == '19'
    assert harness.tool_effects == [(12, 7)]
    followup = model.requests[1].messages
    assert followup[-2].tool_calls[0]['name'] == 'add'
    assert followup[-1].role == 'tool' and followup[-1].tool_results[0]['output'] == 19
    assert [m.role for m in outcome.messages] == ['user', 'assistant', 'tool', 'assistant']


async def test_model_recovery_reuses_completed_tool(harness, monkeypatch):
    from agentweave._runtime.tools import ToolExecutor
    from agentweave.testing import FakeModel, text_turn, tool_turn
    model = FakeModel([tool_turn('add', {'a': 12, 'b': 7}), text_turn('19')])
    await harness.begin(calculator(harness)(model), model)
    original = ToolExecutor.execute

    async def crash_after_commit(self, *args, **kwargs):
        await original(self, *args, **kwargs)
        raise SimulatedCrash  # the tool result is journaled, then the worker dies
    monkeypatch.setattr(ToolExecutor, 'execute', crash_after_commit)
    with pytest.raises(SimulatedCrash):
        await harness.run()
    monkeypatch.undo()
    outcome = await harness.run()  # a restarted worker replays the journal
    assert outcome.status == 'succeeded' and outcome.output == '19'
    assert harness.tool_effects == [(12, 7)]
    assert len(model.requests) == 2  # the completed first turn is not requested again


async def test_invalid_or_unknown_tool_calls_are_reported_back_not_dispatched(harness):
    from agentweave.testing import FakeModel, text_turn, tool_turn
    model = FakeModel([tool_turn('add', {'a': 'twelve', 'b': 7}), tool_turn('delete_everything', {}),
                       text_turn('done')])
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run()
    assert outcome.status == 'succeeded' and harness.tool_effects == []
    assert model.requests[1].messages[-1].tool_results[0]['status'] == 'invalid'
    assert model.requests[2].messages[-1].tool_results[0]['status'] == 'invalid'


async def test_all_calls_of_a_turn_are_validated_before_any_dispatch(harness):
    from agentweave.messages import Message
    from agentweave.models import ModelTurn, ProposedToolCall
    from agentweave.testing import FakeModel, text_turn
    calls = [ProposedToolCall(provider_call_id='c1', name='add', arguments={'a': 1, 'b': 2}),
             ProposedToolCall(provider_call_id='c2', name='add', arguments={'a': 'x', 'b': 2})]
    turn = ModelTurn(messages=[Message(role='assistant', tool_calls=[c.model_dump() for c in calls])],
                     tool_calls=calls, usage={'input_tokens': 1, 'output_tokens': 1})
    model = FakeModel([turn, text_turn('ok')])
    await harness.begin(calculator(harness)(model), model)
    await harness.run()
    assert harness.tool_effects == []  # the valid sibling did not run either
    statuses = [r['status'] for r in model.requests[1].messages[-1].tool_results]
    assert statuses == ['skipped', 'invalid']


async def test_unsupported_capability_fails_before_any_request(harness):
    from agentweave.models import ModelCapabilities
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([text_turn('x')], capabilities=ModelCapabilities(tools=False))
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run()
    assert outcome.status == 'failed' and outcome.error['kind'] == 'unsupported'
    assert model.requests == []


async def test_transient_errors_retry_and_every_attempt_is_counted(harness):
    from agentweave.errors import ModelError
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([ModelError('rate_limit', 'slow down', retryable=True), text_turn('ok')])
    await harness.begin(calculator(harness)(model), model)
    assert (await harness.run()).output == 'ok'
    assert [row[0] for row in harness.attempts()] == ['failed', 'completed']


async def test_non_retryable_errors_fail_the_task(harness):
    from agentweave.errors import ModelError
    from agentweave.testing import FakeModel
    model = FakeModel([ModelError('auth', 'bad key', retryable=False)])
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run()
    assert outcome.status == 'failed' and outcome.error['kind'] == 'auth' and len(model.requests) == 1


async def test_a_step_gives_up_after_three_attempts(harness):
    from agentweave.errors import ModelError
    from agentweave.testing import FakeModel
    model = FakeModel([ModelError('unavailable', 'down', retryable=True)] * 3)
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run()
    assert outcome.status == 'failed' and len(model.requests) == 3


async def test_root_call_budget_stops_an_endless_tool_loop(harness):
    from agentweave.testing import FakeModel, tool_turn
    model = FakeModel([tool_turn('add', {'a': 1, 'b': 1}, call_id=f'c{i}') for i in range(5)])
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run(max_model_calls=2)
    assert outcome.status == 'failed' and outcome.error['type'] == 'ModelBudgetExceeded'
    assert len(model.requests) == 2


async def test_token_budget_blocks_the_next_attempt_and_survives_restart(harness):
    from agentweave.models import ModelBudget
    from agentweave.testing import FakeModel, text_turn, tool_turn
    model = FakeModel([tool_turn('add', {'a': 1, 'b': 2}, usage={'input_tokens': 60, 'output_tokens': 40}),
                       text_turn('never')])
    await harness.begin(calculator(harness, budget=ModelBudget(max_tokens=100))(model), model)
    assert (await harness.run()).error['type'] == 'ModelBudgetExceeded'
    assert (await harness.run()).error['type'] == 'ModelBudgetExceeded'  # restart keeps the count
    assert len(model.requests) == 1


async def test_known_cost_at_the_limit_blocks_the_next_attempt(harness):
    from agentweave.models import ModelBudget, ModelPricing
    from agentweave.testing import FakeModel, text_turn, tool_turn
    pricing = ModelPricing(revision='test-1', input_per_million_usd='1', output_per_million_usd='0')
    model = FakeModel([tool_turn('add', {'a': 1, 'b': 2}, usage={'input_tokens': 10_000, 'output_tokens': 0}),
                       text_turn('never')])
    await harness.begin(calculator(harness, budget=ModelBudget(max_cost_usd='0.01', pricing=pricing))(model), model)
    assert (await harness.run()).error['type'] == 'ModelBudgetExceeded'
    assert harness.attempts()[0][4] == '0.01'


async def test_unknown_usage_blocks_only_an_enabled_budget(harness):
    from agentweave.models import ModelBudget
    from agentweave.testing import FakeModel, text_turn, tool_turn
    unknown = {'input_tokens': None, 'output_tokens': None}
    model = FakeModel([tool_turn('add', {'a': 1, 'b': 2}, usage=unknown), text_turn('ok')])
    await harness.begin(calculator(harness, budget=ModelBudget(max_tokens=1000))(model), model)
    assert (await harness.run()).error['type'] == 'ModelBudgetExceeded'
    model = FakeModel([tool_turn('add', {'a': 1, 'b': 2}, usage=unknown), text_turn('ok')])
    await harness.begin(calculator(harness)(model), model)
    assert (await harness.run()).output == 'ok'
    assert harness.attempts()[0][3] == 0  # still visibly unknown, never counted as zero


async def test_duplicate_finish_is_not_counted_twice(harness):
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([])
    await harness.begin(calculator(harness)(model), model)
    attempt = await harness.store.reserve_model_attempt(harness.lease, step=0, request_hash='h',
        pool_name='custom', pool_limit=4, budget=None, max_calls=20, instance_id=INSTANCE)
    turn = text_turn('x', usage={'input_tokens': 5, 'output_tokens': 5})
    for _ in range(2):
        await harness.store.finish_model_attempt(attempt.attempt_id, turn, input_tokens=5, output_tokens=5)
    usage = await harness.store.model_usage(harness.task.root_task_id)
    assert usage.calls == 1 and usage.tokens == 10 and usage.unknown_usage == 0


async def test_shared_pool_never_runs_two_requests_at_once(harness, envelope_factory):
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([text_turn('a'), text_turn('b')], latency=0.05)
    agent = calculator(harness)(model)
    first = await ModelHarness(harness.store, harness.clock, envelope_factory).begin(agent, model)
    second = await ModelHarness(harness.store, harness.clock, envelope_factory).begin(agent, model)

    async def real_sleep(seconds):
        await asyncio.sleep(0.01)
    first.sleep = second.sleep = real_sleep
    outcomes = await asyncio.gather(first.run(pool_limit=1), second.run(pool_limit=1))
    assert sorted(o.output for o in outcomes) == ['a', 'b']
    assert model.peak_concurrency == 1


async def test_late_response_after_expiry_is_kept_but_its_tool_never_runs(harness):
    from agentweave.testing import FakeModel, tool_turn
    model = FakeModel([tool_turn('add', {'a': 12, 'b': 7})],
                      on_generate=lambda request: harness.clock.advance(10))
    agent = calculator(harness)(model)
    await harness.begin(agent, model, expires_at=harness.clock.now() + timedelta(seconds=5))
    outcome = await harness.run()
    assert outcome.status == 'expired' and harness.tool_effects == []
    assert harness.attempts()[0][0] == 'completed'  # the late answer is journaled


async def test_worker_runs_the_managed_loop_end_to_end(runtime_harness):
    from agentweave import Agent, Model, tool
    from agentweave.testing import FakeModel, text_turn, tool_turn
    effects = []

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        effects.append((a, b))
        return a + b
    model = FakeModel([tool_turn('add', {'a': 12, 'b': 7}), text_turn('12 + 7 = 19')])
    agent = Agent('assistant', tools=[add], model=Model.custom(model), memory='stateful')
    await runtime_harness.start(agent)
    session = str(uuid4())
    result = await runtime_harness.ask('assistant', '12 + 7?', session_id=session)
    assert result.status == 'succeeded' and result.output == '12 + 7 = 19' and effects == [(12, 7)]
    progress = await runtime_harness.store.read_progress(result.task_id)
    assert ''.join(e.payload['text'] for e in progress) == '12 + 7 = 19'
    state, history = await runtime_harness.store.load_session('assistant', session)
    assert [m.role for m in history] == ['user', 'assistant', 'tool', 'assistant']


async def test_a_stalled_model_stream_stops_at_the_task_deadline(harness):
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([text_turn('too late')], latency=5)
    await harness.begin(calculator(harness)(model), model, deadline=harness.clock.now() + timedelta(seconds=0.2))
    outcome = await asyncio.wait_for(harness.run(), 3)
    assert outcome.status == 'expired'
    assert harness.attempts()[0][0] == 'unknown'  # the stalled request may have been billed


async def test_transient_retries_stop_at_the_task_deadline(harness):
    from agentweave.errors import ModelError
    from agentweave.testing import FakeModel
    model = FakeModel([ModelError('rate_limit', 'slow down', retryable=True)] * 3)
    await harness.begin(calculator(harness)(model), model)
    harness.sleep = lambda seconds: _advance(harness.clock, 200)  # the backoff outlives the deadline
    outcome = await harness.run()
    assert outcome.status == 'expired' and len(model.requests) == 1


async def _advance(clock, seconds):
    clock.advance(seconds)


async def test_an_empty_final_turn_is_a_failure(harness):
    from agentweave.messages import Message
    from agentweave.models import ModelTurn
    from agentweave.testing import FakeModel
    model = FakeModel([ModelTurn(messages=[Message(role='assistant')], usage={'input_tokens': 1, 'output_tokens': 0})])
    await harness.begin(calculator(harness)(model), model)
    outcome = await harness.run()
    assert outcome.status == 'failed' and outcome.error['type'] == 'EmptyModelResponse'


async def test_a_journal_conflict_on_replay_fails_the_task_clearly(harness, monkeypatch):
    from agentweave.errors import StoreConflict
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([text_turn('x')])
    await harness.begin(calculator(harness)(model), model)

    async def conflict(*args, **kwargs):
        raise StoreConflict('a completed model step differs from the replayed request')
    monkeypatch.setattr(harness.store, 'reserve_model_attempt', conflict)
    outcome = await harness.run()
    assert outcome.status == 'failed' and outcome.error['type'] == 'StoreConflict'


async def test_stopping_the_worker_does_not_leave_the_loop_running(runtime_harness):
    from agentweave import Agent, Model
    from agentweave.testing import FakeModel, text_turn
    model = FakeModel([text_turn('slow')], latency=5)
    worker = await runtime_harness.start(Agent('slowpoke', model=Model.custom(model)))
    await runtime_harness.client.agent('slowpoke').submit('hi')

    async def requested():
        return bool(model.requests)
    await eventually(requested)
    await worker.stop()
    await asyncio.sleep(.1)
    assert model.active == 0  # the in-flight request was cancelled with its worker


async def test_duplicate_provider_call_ids_in_one_turn_run_nothing(harness):
    from agentweave.messages import Message
    from agentweave.models import ModelTurn, ProposedToolCall
    from agentweave.testing import FakeModel, text_turn
    calls = [ProposedToolCall(provider_call_id='same', name='add', arguments={'a': 5, 'b': 0}),
             ProposedToolCall(provider_call_id='same', name='add', arguments={'a': 999, 'b': 0})]
    turn = ModelTurn(messages=[Message(role='assistant', tool_calls=[c.model_dump() for c in calls])],
                     tool_calls=calls, usage={'input_tokens': 1, 'output_tokens': 1})
    model = FakeModel([turn, text_turn('ok')])
    await harness.begin(calculator(harness)(model), model)
    await harness.run()
    assert harness.tool_effects == []
    errors = [r['error']['type'] for r in model.requests[1].messages[-1].tool_results]
    assert errors == ['DuplicateCallId', 'DuplicateCallId']


async def test_model_prevalidation_accepts_json_forms_of_typed_arguments(harness):
    from datetime import datetime
    from agentweave import Agent, Model, tool
    from agentweave.testing import FakeModel, text_turn, tool_turn

    @tool
    def book(slot: tuple[int, int], at: datetime) -> str:
        """Book a slot."""
        harness.tool_effects.append(slot)
        return 'booked'
    model = FakeModel([tool_turn('book', {'slot': [1, 2], 'at': '2025-01-15T09:30:00Z'}), text_turn('done')])
    agent = Agent('assistant', tools=[book], model=Model.custom(model))
    await harness.begin(agent, model)
    assert (await harness.run()).output == 'done'
    assert harness.tool_effects == [(1, 2)]
