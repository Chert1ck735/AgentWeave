"""Operator repair of outcome_unknown tasks: versioned, audited, one continuation, nothing repeated."""
import asyncio
import json
import sqlite3
from datetime import timedelta
import pytest
from agentweave._records import StepCommit

INSTANCE = '00000000-0000-0000-0000-0000000000c1'


class Accepted:
    def __init__(self, accepted):
        self.accepted = accepted


class RepairHarness:
    """A managed loop whose tool's effect became unknown, committed as the worker would."""

    def __init__(self, store, store_factory, clock, envelope_factory, tmp_path):
        self.store, self.store_factory, self.clock = store, store_factory, clock
        self.envelope_factory, self.tmp_path = envelope_factory, tmp_path
        self.effects, self.fail_next = [], True
        self.task = self.model = self.agent = None

    def build(self):
        from agentweave import Agent, Model, tool
        from agentweave.testing import FakeModel, text_turn, tool_turn
        harness = self

        @tool
        def charge(amount: float, api_token: str) -> float:
            """Charge a card; not safe to repeat."""
            harness.effects.append(amount)
            if harness.fail_next:
                harness.fail_next = False
                raise RuntimeError('the connection dropped after the charge was sent')
            return amount
        self.model = FakeModel([tool_turn('charge', {'amount': 12.5, 'api_token': 'tok-secret'}), text_turn('charged')])
        self.agent = Agent('assistant', tools=[charge], model=Model.custom(self.model))

    async def unknown(self):
        """Admit, run the loop to its unknown tool and commit outcome_unknown."""
        self.build()
        event = self.envelope_factory(recipient='assistant', payload='charge 12.5')
        await self.store.admit(event, 'def-v1')
        outcome = await self._run(event.task_id)
        assert outcome.status == 'outcome_unknown'
        self.task = await self.store.get_task(event.task_id)
        assert self.task.status == 'outcome_unknown'
        return self.task

    async def _run(self, task_id):
        from agentweave._runtime.steps import ModelLoop
        from agentweave._runtime.tools import ToolExecutor
        lease = await self.store.claim(task_id, INSTANCE, 30)
        task = await self.store.get_task(task_id)
        tools = ToolExecutor(self.agent, self.store, instance_id=INSTANCE, clock=self.clock)
        loop = ModelLoop(self.agent, self.store, self.model, instance_id=INSTANCE, clock=self.clock, tools=tools)
        try:
            outcome = await loop.run(task, lease)
        finally:
            loop.close()
        await self.store.commit_step(lease, StepCommit(step_id='loop.' + str(lease.token), expected_version=task.version,
                                                       status=outcome.status, result=outcome.output, error=outcome.error))
        return outcome

    async def resume(self):
        """What a worker's recovery does with an accepted task."""
        return await self._run(self.task.task_id)

    async def unknown_step(self):
        from agentweave._runtime.maintenance import describe
        described = await describe(self.store, self.task.task_id)
        [step] = [s for s in described['steps'] if s['status'] == 'outcome_unknown']
        return step

    async def concurrent_resolve_and_retry(self):
        from agentweave._runtime.maintenance import resolve, retry
        from agentweave.errors import VersionConflict
        await self.unknown()
        step = await self.unknown_step()
        other = await self.store_factory(self.tmp_path / 'state.sqlite3')  # a second operator process

        async def attempt(action):
            try:
                await action
                return Accepted(True)
            except VersionConflict:
                return Accepted(False)
        return await asyncio.gather(
            attempt(resolve(self.store, self.task.task_id, step['step_id'], self.task.version, result=12.5)),
            attempt(retry(other, self.task.task_id, step['step_id'], self.task.version, allow_unknown=True)))

    @property
    def continuation_count(self):
        return len(self.actions(self.task.task_id, kinds=('resolve', 'retry')))

    def actions(self, task_id, kinds=None):
        with sqlite3.connect(self.store.path) as connection:
            rows = connection.execute('SELECT action, step_id, expected_version, details FROM operator_actions '
                                      'WHERE task_id=? ORDER BY id', (task_id,)).fetchall()
        return [row for row in rows if kinds is None or row[0] in kinds]


@pytest.fixture
def repair_harness(store, store_factory, fake_clock, envelope_factory, tmp_path):
    return RepairHarness(store, store_factory, fake_clock, envelope_factory, tmp_path)


async def test_two_operators_create_one_continuation(repair_harness):
    results = await repair_harness.concurrent_resolve_and_retry()
    assert sum(r.accepted for r in results) == 1
    assert repair_harness.continuation_count == 1


async def test_the_unknown_step_is_shown_with_its_kind_and_without_secrets(repair_harness):
    from agentweave._runtime.maintenance import describe
    await repair_harness.unknown()
    step = await repair_harness.unknown_step()
    assert (step['kind'], step['tool_id']) == ('tool', 'charge')
    assert step['arguments'] == {'amount': 12.5, 'api_token': '***'}
    assert 'tok-secret' not in json.dumps(await describe(repair_harness.store, repair_harness.task.task_id))


async def test_resolving_a_tool_checks_its_schema_and_resumes_the_loop_without_repeating_it(repair_harness):
    from agentweave._runtime.maintenance import resolve
    from agentweave.errors import OperatorActionRefused
    task = await repair_harness.unknown()
    step = await repair_harness.unknown_step()
    with pytest.raises(OperatorActionRefused, match='schema'):
        await resolve(repair_harness.store, task.task_id, step['step_id'], task.version, result='twelve')
    assert (await repair_harness.store.get_task(task.task_id)).status == 'outcome_unknown'
    resumed = await resolve(repair_harness.store, task.task_id, step['step_id'], task.version, result=12.5)
    assert resumed.status == 'accepted' and resumed.version == task.version + 1
    outcome = await repair_harness.resume()
    assert outcome.status == 'succeeded' and outcome.output == 'charged'
    assert repair_harness.effects == [12.5]  # the charge was not sent again
    assert len(repair_harness.model.requests) == 2  # the completed first turn was replayed, not requested
    assert repair_harness.model.requests[1].messages[-1].tool_results[0]['output'] == 12.5


async def test_a_forced_retry_repeats_only_the_unknown_step(repair_harness):
    from agentweave._runtime.maintenance import retry
    from agentweave.errors import OperatorActionRefused
    task = await repair_harness.unknown()
    step = await repair_harness.unknown_step()
    with pytest.raises(OperatorActionRefused, match='allow'):
        await retry(repair_harness.store, task.task_id, step['step_id'], task.version, allow_unknown=False)
    await retry(repair_harness.store, task.task_id, step['step_id'], task.version, allow_unknown=True)
    outcome = await repair_harness.resume()
    assert outcome.status == 'succeeded'
    assert repair_harness.effects == [12.5, 12.5]  # the operator chose to send it again, once
    assert len(repair_harness.model.requests) == 2


async def test_an_expired_deadline_needs_an_explicit_new_one(repair_harness):
    from agentweave._runtime.maintenance import resolve
    from agentweave.errors import OperatorActionRefused
    task = await repair_harness.unknown()
    step = await repair_harness.unknown_step()
    repair_harness.clock.advance(600)
    with pytest.raises(OperatorActionRefused, match='deadline'):
        await resolve(repair_harness.store, task.task_id, step['step_id'], task.version, result=12.5)
    with pytest.raises(OperatorActionRefused, match='future'):
        await resolve(repair_harness.store, task.task_id, step['step_id'], task.version, result=12.5,
                      new_deadline=repair_harness.clock.now() - timedelta(seconds=1))
    deadline = repair_harness.clock.now() + timedelta(seconds=120)
    resumed = await resolve(repair_harness.store, task.task_id, step['step_id'], task.version, result=12.5,
                            new_deadline=deadline)
    assert resumed.deadline == deadline
    assert (await repair_harness.store.get_input_envelope(task.task_id)).deadline == deadline
    outcome = await repair_harness.resume()
    assert outcome.status == 'succeeded' and repair_harness.effects == [12.5]


async def test_a_stale_version_is_a_conflict_and_every_action_is_audited(repair_harness):
    from agentweave._runtime.maintenance import fail, resolve
    from agentweave.errors import VersionConflict
    task = await repair_harness.unknown()
    step = await repair_harness.unknown_step()
    with pytest.raises(VersionConflict):
        await resolve(repair_harness.store, task.task_id, step['step_id'], task.version - 1, result=12.5)
    failed = await fail(repair_harness.store, task.task_id, task.version, reason='refunded by hand')
    assert failed.status == 'failed' and failed.error == {'type': 'OperatorFailed', 'message': 'refunded by hand'}
    with pytest.raises(VersionConflict):  # a finished task has no unknown step left
        await resolve(repair_harness.store, task.task_id, step['step_id'], failed.version, result=12.5)
    [(action, step_id, expected, details)] = repair_harness.actions(task.task_id)
    assert (action, step_id, expected) == ('fail', None, task.version)
    assert json.loads(details) == {'reason': 'refunded by hand'}


async def _unknown_handler(store, clock, envelope_factory, agent='counter'):
    """A non-retry-safe Python handler interrupted mid-run: its whole effect is unknown."""
    event = envelope_factory(recipient=agent)
    await store.admit(event, 'def-v1')
    lease = await store.claim_runtime(event.task_id, INSTANCE, 5, concurrency=1)
    assert await store.start_handler(lease, retry_safe=False)
    clock.advance(10)  # the worker died; its lease ran out
    lease = await store.claim_runtime(event.task_id, INSTANCE, 5, concurrency=1)
    assert not await store.start_handler(lease, retry_safe=False)
    task = await store.get_task(event.task_id)
    assert task.status == 'outcome_unknown'
    return event, task


async def test_resolving_an_opaque_handler_finishes_it_with_an_explicit_state(store, fake_clock, envelope_factory):
    from agentweave._runtime.maintenance import describe, resolve
    event, task = await _unknown_handler(store, fake_clock, envelope_factory)
    [step] = (await describe(store, task.task_id))['steps']
    assert (step['step_id'], step['kind'], step['status']) == ('handler', 'python_handler', 'outcome_unknown')
    done = await resolve(store, task.task_id, 'handler', task.version, result={'count': 3}, state={'count': 3})
    assert (done.status, done.result) == ('succeeded', {'count': 3})
    state, _ = await store.load_session('counter', event.session_id)
    assert state == {'count': 3}
    with sqlite3.connect(store.path) as connection:
        [reply] = connection.execute("SELECT envelope FROM outbox WHERE json_extract(envelope,'$.kind')='reply'").fetchall()
    payload = json.loads(reply[0])
    assert payload['recipient'] == event.sender and payload['payload']['status'] == 'succeeded'


async def test_retrying_an_opaque_handler_starts_it_again_from_the_last_confirmed_state(store, fake_clock,
                                                                                         envelope_factory):
    from agentweave._runtime.maintenance import retry
    _, task = await _unknown_handler(store, fake_clock, envelope_factory)
    resumed = await retry(store, task.task_id, 'handler', task.version, allow_unknown=True)
    assert resumed.status == 'accepted'
    lease = await store.claim_runtime(task.task_id, INSTANCE, 5, concurrency=1)
    assert await store.start_handler(lease, retry_safe=False)  # a fresh attempt, not an unknown one


async def test_cancel_stops_new_steps_but_never_declares_an_unknown_effect_cancelled(store, fake_clock,
                                                                                      envelope_factory):
    from agentweave._runtime.maintenance import cancel
    from agentweave.errors import OperatorActionRefused, TaskCancelled
    from agentweave.messages import ToolCall
    queued = envelope_factory(recipient='counter')
    await store.admit(queued, 'def-v1')
    cancelled = await cancel(store, queued.task_id, 0, reason='not needed')
    assert cancelled.status == 'cancelled'
    running = envelope_factory(recipient='counter')
    await store.admit(running, 'def-v1')
    lease = await store.claim(running.task_id, INSTANCE, 30)
    current = await store.get_task(running.task_id)
    requested = await cancel(store, running.task_id, current.version, reason='stop')
    assert requested.status == 'running'  # the step in flight finishes with its real outcome
    call = ToolCall(tool_call_id='00000000-0000-4000-8000-0000000000f1', tool_id='charge', revision='r1',
                    arguments={}, caller='counter', deadline=running.deadline)
    with pytest.raises(TaskCancelled):
        await store.reserve_tool(call, {}, lease=lease, instance_id=INSTANCE, retry_safe=False, max_attempts=1)
    _, unknown = await _unknown_handler(store, fake_clock, envelope_factory, agent='other')
    with pytest.raises(OperatorActionRefused, match='unknown'):
        await cancel(store, unknown.task_id, unknown.version, reason='stop')


async def test_cancel_of_a_crashed_unsafe_attempt_keeps_the_outcome_unknown(store, fake_clock, envelope_factory):
    """The worker died mid-handler: cancelling must not claim the effect never happened."""
    from agentweave._runtime.maintenance import cancel
    event = envelope_factory(recipient='counter')
    await store.admit(event, 'def-v1')
    lease = await store.claim_runtime(event.task_id, INSTANCE, 5, concurrency=1)
    assert await store.start_handler(lease, retry_safe=False)
    fake_clock.advance(10)
    current = await store.get_task(event.task_id)
    ended = await cancel(store, event.task_id, current.version, reason='stop')
    assert ended.status == 'outcome_unknown'


async def test_a_cancel_noticed_at_claim_still_replies_and_tool_calls_are_cancelled_by_their_caller(
        store, fake_clock, envelope_factory):
    from agentweave._runtime.maintenance import cancel
    from agentweave.errors import OperatorActionRefused
    event = envelope_factory(recipient='counter')
    await store.admit(event, 'def-v1')
    await store.claim(event.task_id, INSTANCE, 5)  # a retry-safe step in flight
    current = await store.get_task(event.task_id)
    await cancel(store, event.task_id, current.version, reason='stop')
    fake_clock.advance(10)  # the step ended without committing; recovery claims again
    assert await store.claim(event.task_id, INSTANCE, 5) is None
    assert (await store.get_task(event.task_id)).status == 'cancelled'
    with sqlite3.connect(store.path) as connection:
        replies = connection.execute("SELECT envelope FROM outbox WHERE json_extract(envelope,'$.kind')='reply'").fetchall()
    assert [json.loads(r[0])['payload']['status'] for r in replies] == ['cancelled']
    call = envelope_factory(kind='tool_call', recipient='weather', sender='counter', session_id=None)
    await store.admit(call, 'def-v1')
    with pytest.raises(OperatorActionRefused, match='caller'):
        await cancel(store, call.task_id, 0, reason='stop')
