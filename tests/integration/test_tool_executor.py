"""Durable Python tool execution: validation, hooks, retries, freshness and crashes."""
import sqlite3
from datetime import timedelta
from uuid import uuid4
import pytest

INSTANCE = '00000000-0000-0000-0000-0000000000a1'


class SimulatedCrash(BaseException):
    """Stands in for a process dying between an external effect and its journal commit."""


class ToolHarness:
    def __init__(self, store, clock, envelope_factory):
        self.store, self.clock, self.envelope_factory = store, clock, envelope_factory
        self.delays, self.effects = [], []
        self.agent = self.task = self.lease = None

    async def begin(self, agent, **event_changes):
        self.agent = agent
        event = self.envelope_factory(recipient=agent.name, **event_changes)
        await self.store.admit(event, 'def-v1')
        self.lease = await self.store.claim(event.task_id, INSTANCE, 30)
        self.task = await self.store.get_task(event.task_id)
        return self

    async def sleep(self, seconds):
        self.delays.append(seconds)
        self.clock.advance(seconds)

    def executor(self):
        from agentweave._runtime.tools import ToolExecutor
        return ToolExecutor(self.agent, self.store, instance_id=INSTANCE, clock=self.clock,
                            sleep=self.sleep, random=lambda: 0.0)

    def call(self, definition, arguments, **changes):
        from agentweave.messages import ToolCall
        values = dict(tool_call_id=str(uuid4()), tool_id=definition.name,
                      revision=definition.schema.revision, arguments=arguments,
                      caller=self.agent.name, deadline=self.task.deadline)
        return ToolCall(**(values | changes))

    async def run(self, call):
        executor = self.executor()
        try:
            return await executor.execute(call, lease=self.lease, task=self.task)
        finally:
            executor.close()

    def journal(self, call):
        with sqlite3.connect(self.store.path) as connection:
            return connection.execute('SELECT status, attempts, arguments FROM tool_executions '
                                      'WHERE tool_call_id=?', (call.tool_call_id,)).fetchone()


@pytest.fixture
def harness(store, fake_clock, envelope_factory):
    return ToolHarness(store, fake_clock, envelope_factory)


def agent_with(*tools, **options):
    from agentweave import Agent
    return Agent('assistant', tools=list(tools), **options)


def counting_tool(harness, *, retry=None, fail_times=0, result=None):
    from agentweave import tool

    @tool(retry=retry)
    def charge(amount: int) -> int:
        """Charge an amount."""
        harness.effects.append(amount)
        if len(harness.effects) <= fail_times:
            raise ConnectionError(f'gateway failure {len(harness.effects)}')
        return amount if result is None else result
    return charge


async def test_successful_result_is_journaled_and_replayed_without_a_second_effect(harness):
    charge = counting_tool(harness)
    await harness.begin(agent_with(charge))
    call = harness.call(charge, {'amount': 5})
    first = await harness.run(call)
    second = await harness.run(call)
    assert first.status == second.status == 'succeeded'
    assert first.output == second.output == 5
    assert harness.effects == [5]
    assert harness.journal(call)[:2] == ('succeeded', 1)


async def test_invalid_arguments_never_reach_the_tool_or_the_journal(harness):
    charge = counting_tool(harness)
    await harness.begin(agent_with(charge))
    call = harness.call(charge, {'amount': 'five'})
    result = await harness.run(call)
    assert result.status == 'invalid' and result.error['type'] == 'ValidationError'
    assert harness.effects == [] and harness.journal(call) is None


async def test_stale_revision_and_unknown_tool_are_invalid(harness):
    charge = counting_tool(harness)
    await harness.begin(agent_with(charge))
    assert (await harness.run(harness.call(charge, {'amount': 1}, revision='old'))).status == 'invalid'
    assert (await harness.run(harness.call(charge, {'amount': 1}, tool_id='missing'))).status == 'invalid'
    assert harness.effects == []


async def test_caller_outside_allowed_callers_is_denied(harness):
    from agentweave import tool

    @tool
    def secret(value: int) -> int:
        harness.effects.append(value)
        return value
    restricted = secret.model_copy(update={'allowed_callers': ('auditor',)})
    await harness.begin(agent_with(restricted))
    result = await harness.run(harness.call(restricted, {'value': 1}))
    assert result.status == 'denied' and harness.effects == []


async def test_unknown_effect_is_not_retried(harness, monkeypatch):
    charge = counting_tool(harness)
    await harness.begin(agent_with(charge))
    call = harness.call(charge, {'amount': 7})

    async def crash(*args, **kwargs):
        raise SimulatedCrash
    monkeypatch.setattr(harness.store, 'finish_tool', crash)
    with pytest.raises(SimulatedCrash):
        await harness.run(call)
    monkeypatch.undo()
    result = await harness.run(call)  # a restarted worker resumes the same call
    assert result.status == 'outcome_unknown'
    assert harness.effects == [7]
    assert harness.journal(call)[0] == 'outcome_unknown'


async def test_safe_tool_resumes_an_interrupted_attempt(harness, monkeypatch):
    charge = counting_tool(harness, retry='safe')
    await harness.begin(agent_with(charge))
    call = harness.call(charge, {'amount': 3})
    original = harness.store.finish_tool

    async def crash_once(*args, **kwargs):
        monkeypatch.setattr(harness.store, 'finish_tool', original)
        raise SimulatedCrash
    monkeypatch.setattr(harness.store, 'finish_tool', crash_once)
    with pytest.raises(SimulatedCrash):
        await harness.run(call)
    result = await harness.run(call)
    assert result.status == 'succeeded' and harness.effects == [3, 3]
    assert harness.journal(call)[:2] == ('succeeded', 2)


async def test_safe_failures_retry_with_bounded_backoff_then_fail(harness):
    from agentweave.definitions import RetryPolicy
    charge = counting_tool(harness, retry=RetryPolicy(safe=True, max_attempts=3, min_delay=1, max_delay=10),
                           fail_times=5)
    await harness.begin(agent_with(charge))
    result = await harness.run(harness.call(charge, {'amount': 2}))
    assert result.status == 'failed'
    assert result.error == {'type': 'ConnectionError', 'message': 'gateway failure 3'}
    assert len(harness.effects) == 3
    assert harness.delays == [1, 2] and all(1 <= d <= 10 for d in harness.delays)


async def test_safe_retry_recovers_after_a_transient_failure(harness):
    charge = counting_tool(harness, retry='safe', fail_times=1)
    await harness.begin(agent_with(charge))
    result = await harness.run(harness.call(charge, {'amount': 4}))
    assert result.status == 'succeeded' and harness.effects == [4, 4]


async def test_unsafe_failure_after_a_possible_effect_is_unknown(harness):
    charge = counting_tool(harness, fail_times=1)
    await harness.begin(agent_with(charge))
    result = await harness.run(harness.call(charge, {'amount': 9}))
    assert result.status == 'outcome_unknown' and harness.effects == [9]
    assert result.error['type'] == 'ConnectionError'


async def test_sync_timeout_is_unknown_and_not_retried(harness):
    import threading
    from agentweave import tool
    from agentweave.definitions import RetryPolicy
    release = threading.Event()

    @tool(retry=RetryPolicy(timeout=.05))
    def slow(value: int) -> int:
        harness.effects.append(value)
        release.wait(5)
        return value
    await harness.begin(agent_with(slow))
    try:
        result = await harness.run(harness.call(slow, {'value': 1}))
    finally:
        release.set()
    assert result.status == 'outcome_unknown' and harness.effects == [1]


async def test_invalid_output_after_the_effect_is_unknown_and_not_retried(harness):
    from agentweave import tool

    @tool(retry='safe')
    def broken(value: int) -> int:
        harness.effects.append(value)
        return 'not an int'
    await harness.begin(agent_with(broken))
    result = await harness.run(harness.call(broken, {'value': 1}))
    # The effect ran but its result is unusable: report it as unknown, never as a clean failure.
    assert result.status == 'outcome_unknown' and harness.effects == [1]
    assert result.error['type'] == 'InvalidToolOutput'


async def test_before_hook_deny_consumes_no_attempt(harness):
    from agentweave.hooks import Deny
    charge = counting_tool(harness)
    agent = agent_with(charge)
    agent.before_tool(lambda call, ctx: Deny('limit exceeded'))
    await harness.begin(agent)
    call = harness.call(charge, {'amount': 1})
    result = await harness.run(call)
    assert result.status == 'denied' and result.error['message'] == 'limit exceeded'
    assert harness.effects == [] and harness.journal(call) is None


async def test_before_hook_changes_are_revalidated_and_journaled(harness):
    charge = counting_tool(harness)
    agent = agent_with(charge)

    @agent.before_tool
    async def cap(call, ctx):
        return call.model_copy(update={'arguments': {'amount': min(call.arguments['amount'], 10)}})
    await harness.begin(agent)
    call = harness.call(charge, {'amount': 500})
    result = await harness.run(call)
    assert result.status == 'succeeded' and harness.effects == [10]
    assert '"amount":10' in harness.journal(call)[2].replace(' ', '')


async def test_before_hook_invalid_change_is_rejected(harness):
    charge = counting_tool(harness)
    agent = agent_with(charge)
    agent.before_tool(lambda call, ctx: call.model_copy(update={'arguments': {'amount': 'all'}}))
    await harness.begin(agent)
    assert (await harness.run(harness.call(charge, {'amount': 1}))).status == 'invalid'
    assert harness.effects == []


@pytest.mark.parametrize('field', ['tool_call_id', 'caller', 'deadline', 'expires_at'])
async def test_before_hook_cannot_change_protected_fields(harness, field):
    charge = counting_tool(harness)
    agent = agent_with(charge)
    changes = {'tool_call_id': str(uuid4()), 'caller': 'impostor',
               'deadline': harness.clock.now() + timedelta(days=1),
               'expires_at': harness.clock.now() + timedelta(days=1)}
    agent.before_tool(lambda call, ctx: call.model_copy(update={field: changes[field]}))
    await harness.begin(agent)
    result = await harness.run(harness.call(charge, {'amount': 1}))
    assert result.status == 'denied' and harness.effects == []


async def test_after_hook_failure_is_journaled_and_never_repeats_the_effect(harness):
    charge = counting_tool(harness)
    agent = agent_with(charge)
    calls = []

    @agent.after_tool
    def audit(call, result, ctx):
        calls.append(result)
        raise RuntimeError('audit sink down')
    await harness.begin(agent)
    call = harness.call(charge, {'amount': 6})
    first = await harness.run(call)
    second = await harness.run(call)
    assert first.status == second.status == 'failed'
    assert first.error == {'type': 'RuntimeError', 'message': 'audit sink down'}
    assert harness.effects == [6] and calls == [6]
    assert harness.journal(call)[0] == 'succeeded'  # the external result itself


async def test_after_hook_transforms_the_returned_result(harness):
    charge = counting_tool(harness)
    agent = agent_with(charge)
    agent.after_tool(lambda call, result, ctx: {'charged': result})
    await harness.begin(agent)
    call = harness.call(charge, {'amount': 2})
    assert (await harness.run(call)).output == {'charged': 2}
    assert (await harness.run(call)).output == {'charged': 2}
    assert harness.effects == [2]


async def test_expired_input_starts_no_new_effect(harness):
    charge = counting_tool(harness)
    await harness.begin(agent_with(charge))
    call = harness.call(charge, {'amount': 1}, expires_at=harness.clock.now() + timedelta(seconds=5))
    harness.clock.advance(6)  # freshness passed while the model was thinking
    result = await harness.run(call)
    assert result.status == 'expired' and harness.effects == []


async def test_safe_retry_stops_when_freshness_runs_out(harness):
    from agentweave.definitions import RetryPolicy
    charge = counting_tool(harness, retry=RetryPolicy(safe=True, max_attempts=3, min_delay=6, max_delay=6),
                           fail_times=5)
    await harness.begin(agent_with(charge))
    call = harness.call(charge, {'amount': 1}, expires_at=harness.clock.now() + timedelta(seconds=5))
    result = await harness.run(call)
    assert result.status == 'failed' and harness.effects == [1]
    assert harness.delays == []  # a retry that cannot start fresh is not attempted


async def test_tool_context_carries_call_identity_and_attempt(harness):
    from agentweave import tool
    from agentweave.context import ToolContext
    seen = []

    @tool
    async def stamp(value: int, ctx: ToolContext) -> int:
        seen.append((ctx.tool_call_id, ctx.task_id, ctx.caller, ctx.attempt))
        return value
    await harness.begin(agent_with(stamp))
    call = harness.call(stamp, {'value': 1})
    assert (await harness.run(call)).status == 'succeeded'
    assert seen == [(call.tool_call_id, harness.task.task_id, 'assistant', 1)]


async def test_stale_task_lease_cannot_start_an_effect(harness):
    from agentweave.errors import LeaseLost
    charge = counting_tool(harness)
    await harness.begin(agent_with(charge))
    harness.clock.advance(31)  # the task lease expired; another worker may own it now
    with pytest.raises(LeaseLost):
        await harness.run(harness.call(charge, {'amount': 1}))
    assert harness.effects == []


async def test_a_call_started_before_expiry_keeps_its_response(harness):
    import asyncio
    from agentweave import tool

    @tool
    async def slow(value: int) -> int:
        harness.effects.append(value)
        await asyncio.sleep(.2)  # finishes after expires_at; it was sent while fresh
        return value
    await harness.begin(agent_with(slow))
    import datetime
    real_now = harness.clock.now

    class Ticking:
        """A clock that moves with real time, so expiry passes mid-call."""
        start = datetime.datetime.now(datetime.timezone.utc)
        def now(self):
            return real_now() + (datetime.datetime.now(datetime.timezone.utc) - self.start)
    harness.clock = Ticking()
    call = harness.call(slow, {'value': 1}, expires_at=real_now() + timedelta(milliseconds=50))
    result = await harness.run(call)
    assert result.status == 'succeeded' and result.output == 1 and harness.effects == [1]


async def test_resume_rechecks_revision_permission_and_arguments(harness, monkeypatch):
    from agentweave import tool
    charge = counting_tool(harness, retry='safe')
    await harness.begin(agent_with(charge))
    call = harness.call(charge, {'amount': 3})
    original = harness.store.finish_tool

    async def crash_once(*args, **kwargs):
        monkeypatch.setattr(harness.store, 'finish_tool', original)
        raise SimulatedCrash
    monkeypatch.setattr(harness.store, 'finish_tool', crash_once)
    with pytest.raises(SimulatedCrash):
        await harness.run(call)

    @tool(retry='safe')
    def charge(amount: int) -> int:  # redeployed with a different contract
        """Charge an amount in cents."""
        harness.effects.append(amount)
        return amount
    harness.agent = agent_with(charge)
    assert (await harness.run(call)).status == 'invalid'
    revoked = charge.model_copy(update={'allowed_callers': ('auditor',)})
    harness.agent = agent_with(revoked.model_copy(update={'schema': counting_tool(harness).schema}))
    assert (await harness.run(call)).status in ('invalid', 'denied')
    assert harness.effects == [3]  # nothing ran after the redeploy
    assert harness.journal(call)[0] == 'running'  # the row is left for an operator or a matching worker


async def test_expired_call_runs_no_hooks(harness):
    charge = counting_tool(harness)
    agent = agent_with(charge)
    seen = []
    agent.before_tool(lambda call, ctx: seen.append(call) or call)
    await harness.begin(agent)
    call = harness.call(charge, {'amount': 1}, expires_at=harness.clock.now() + timedelta(seconds=1))
    harness.clock.advance(2)
    assert (await harness.run(call)).status == 'expired'
    assert seen == [] and harness.effects == []


def test_backoff_jitter_spreads_within_the_policy_bounds(store, fake_clock):
    from agentweave._runtime.tools import ToolExecutor
    from agentweave.definitions import RetryPolicy
    policy = RetryPolicy(safe=True, min_delay=1, max_delay=10)
    def delays(value):
        executor = ToolExecutor(agent_with(), store, instance_id=INSTANCE, clock=fake_clock, random=lambda: value)
        try:
            return [executor._backoff(policy, attempt) for attempt in (1, 2, 3, 4, 5)]
        finally:
            executor.close()
    assert delays(0.0) == [1, 2, 4, 8, 10]
    assert delays(1.0) == [2, 4, 8, 10, 10]
    assert all(1 <= d <= 10 for d in delays(0.5)) and delays(0.5)[0] > 1


async def test_a_settled_call_id_reused_with_other_arguments_is_rejected(harness):
    charge = counting_tool(harness)
    await harness.begin(agent_with(charge))
    first = harness.call(charge, {'amount': 5})
    assert (await harness.run(first)).output == 5
    conflicting = first.model_copy(update={'arguments': {'amount': 999}})
    result = await harness.run(conflicting)
    assert result.status == 'invalid' and result.error['type'] == 'ConflictingCallId'
    assert harness.effects == [5]


async def test_a_pending_call_id_reused_with_other_arguments_is_rejected(harness, monkeypatch):
    charge = counting_tool(harness, retry='safe')
    await harness.begin(agent_with(charge))
    first = harness.call(charge, {'amount': 5})

    async def crash(*args, **kwargs):
        raise SimulatedCrash
    monkeypatch.setattr(harness.store, 'finish_tool', crash)
    with pytest.raises(SimulatedCrash):
        await harness.run(first)
    monkeypatch.undo()
    result = await harness.run(first.model_copy(update={'arguments': {'amount': 999}}))
    assert result.status == 'invalid' and result.error['type'] == 'ConflictingCallId'
    assert harness.effects == [5]


async def test_json_forms_of_typed_arguments_are_accepted(harness):
    from datetime import datetime, timezone
    from agentweave import tool
    seen = []

    @tool
    def book(slot: tuple[int, int], at: datetime) -> str:
        """Book a slot."""
        seen.append((slot, at))
        harness.effects.append(slot)
        return at.isoformat()
    await harness.begin(agent_with(book))
    result = await harness.run(harness.call(book, {'slot': [1, 2], 'at': '2025-01-15T09:30:00Z'}))
    assert result.status == 'succeeded'
    assert seen == [((1, 2), datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc))]


async def test_after_hook_sees_the_arguments_that_actually_ran(harness):
    charge = counting_tool(harness)
    agent = agent_with(charge)
    seen = []

    @agent.before_tool
    def cap(call, ctx):
        return call.model_copy(update={'arguments': {'amount': min(call.arguments['amount'], 10)}})

    @agent.after_tool
    def audit(call, result, ctx):
        seen.append((call.arguments, ctx.requested_arguments, result))
        return result
    await harness.begin(agent)
    await harness.run(harness.call(charge, {'amount': 500}))
    assert seen == [({'amount': 10}, {'amount': 500}, 10)]


async def test_a_replaced_executor_shares_the_busy_tool_pool(harness):
    import asyncio
    import threading
    from agentweave import tool
    from agentweave._runtime.tools import ToolExecutor
    from agentweave.definitions import RetryPolicy
    release, active, peak = threading.Event(), [0], [0]

    @tool(retry=RetryPolicy(timeout=.05))
    def slow(value: int) -> int:
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        try:
            if value == 1:
                release.wait(5)  # outlives its timeout and its executor
            return value
        finally:
            active[0] -= 1
    await harness.begin(agent_with(slow))
    owner = ('network', 'assistant', 'tools')
    first = ToolExecutor(harness.agent, harness.store, instance_id=INSTANCE, clock=harness.clock, owner=owner)
    assert (await first.execute(harness.call(slow, {'value': 1}), lease=harness.lease, task=harness.task)).status == 'outcome_unknown'
    first.close()  # the worker was replaced while its tool thread still runs
    second = ToolExecutor(harness.agent, harness.store, instance_id=INSTANCE, clock=harness.clock, owner=owner)
    patient = harness.call(slow, {'value': 2})
    second_slow = slow.model_copy(update={'retry': RetryPolicy(timeout=5)})
    second.tools['slow'] = second_slow
    pending = asyncio.create_task(second.execute(patient, lease=harness.lease, task=harness.task))
    await asyncio.sleep(.2)
    assert not pending.done() and peak[0] == 1  # the old thread still holds the only slot
    release.set()
    assert (await asyncio.wait_for(pending, 5)).status == 'succeeded'
    second.close()
    assert peak[0] == 1


async def test_a_tool_timing_out_while_waiting_for_a_slot_expires_without_an_effect(harness):
    import threading
    from agentweave import tool
    from agentweave._runtime.tools import ToolExecutor
    from agentweave.definitions import RetryPolicy
    release, effects = threading.Event(), []

    @tool(retry=RetryPolicy(timeout=.05))
    def slow(value: int) -> int:
        effects.append(value)
        if value == 1:
            release.wait(5)  # keeps the only thread after its own timeout
        return value
    await harness.begin(agent_with(slow))
    executor = ToolExecutor(harness.agent, harness.store, instance_id=INSTANCE, clock=harness.clock)
    try:
        held = harness.call(slow, {'value': 1})
        assert (await executor.execute(held, lease=harness.lease, task=harness.task)).status == 'outcome_unknown'
        queued = harness.call(slow, {'value': 2})
        result = await executor.execute(queued, lease=harness.lease, task=harness.task)
        # The code never started, so this is not an unknown effect.
        assert result.status == 'expired' and effects == [1]
        assert harness.journal(queued)[0] == 'expired'
    finally:
        release.set()
        executor.close()
