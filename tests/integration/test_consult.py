"""Consulting peers: durable waits that free execution slots, shared root limits, cycles, recovery."""
import asyncio
import sqlite3
from tests.integration.test_python_runtime import eventually, runtime_harness  # noqa: F401

FREE = {'input_tokens': 0, 'output_tokens': 0}


def rows(store, sql, values=()):
    with sqlite3.connect(store.path) as connection:
        return connection.execute(sql, values).fetchall()


def consult_turn(agent, value, call_id='call_1', usage=FREE):
    from agentweave.testing import tool_turn
    return tool_turn('consult_agent', {'agent': agent, 'input': value}, call_id=call_id, usage=usage)


def assistant(model, name='assistant', **options):
    from agentweave import Agent, Model, Peers
    options.setdefault('peers', Peers.discover())
    return Agent(name, instructions='Ask colleagues when needed.', model=Model.custom(model), **options)


def gated_analyst(gate, seen, name='analyst', **options):
    from agentweave import Agent
    agent = Agent(name, **options)

    @agent.on_request
    async def answer(ctx, request):
        seen.append((request.input, len(request.history), ctx.session_id))
        if gate is not None:
            await gate.wait()
        return {'answer': request.input}
    return agent


async def _status(h, task_id, status):
    task = await h.store.get_task(task_id)
    return task is not None and task.status == status


async def test_a_managed_loop_consults_a_peer_and_continues_with_its_answer(runtime_harness):
    from agentweave.testing import FakeModel, text_turn
    h, seen = runtime_harness, []
    model = FakeModel([consult_turn('analyst', '2+2'), text_turn('analyst says 4')])
    await h.start(gated_analyst(None, seen))
    await h.start(assistant(model))
    result = await h.ask('assistant', 'what is 2+2?')
    assert (result.status, result.output) == ('succeeded', 'analyst says 4')
    offered = {t.tool_id: t for t in model.requests[0].tools}
    assert offered['consult_agent'].input_schema['properties']['agent']['enum'] == ['analyst']
    assert model.requests[1].messages[-1].tool_results[0]['output'] == {'answer': '2+2'}
    assert [entry[:2] for entry in seen] == [('2+2', 0)]  # only the explicit request, no parent history


async def test_waiting_parent_releases_capacity(runtime_harness):
    from agentweave.messages import decode
    from agentweave.testing import FakeModel, text_turn
    h, seen, gate = runtime_harness, [], asyncio.Event()
    model = FakeModel([consult_turn('analyst', 'slow'), text_turn('second'), text_turn('first done')])
    await h.start(gated_analyst(gate, seen))
    await h.start(assistant(model, concurrency=1))
    first = await h.client.agent('assistant').submit('first')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    assert rows(h.store, "SELECT COUNT(*) FROM tasks WHERE agent_id='assistant' AND status='running'") == [(0,)]
    assert (await h.ask('assistant', 'second')).output == 'second'  # the only slot was free meanwhile
    gate.set()
    result = await asyncio.wait_for(h.client.result(first.task_id), 20)
    assert (result.status, result.output) == ('succeeded', 'first done')
    # The child's reply delivered again resumes nothing a second time.
    [(raw,)] = rows(h.store, "SELECT envelope FROM outbox WHERE json_extract(envelope,'$.kind')='reply' "
                             "AND json_extract(envelope,'$.recipient')='assistant'")
    for _ in range(2):
        await h.bus.publish(decode(raw.encode()))
    await asyncio.sleep(.5)
    assert len(model.requests) == 3
    assert rows(h.store, "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='resumed'",
                (first.task_id,)) == [(1,)]


async def test_consulted_peers_never_see_old_sessions(runtime_harness):
    from agentweave.testing import FakeModel, text_turn
    h, seen = runtime_harness, []
    model = FakeModel([consult_turn('analyst', 'one'), text_turn('a'), consult_turn('analyst', 'two'), text_turn('b')])
    await h.start(gated_analyst(None, seen, memory='stateful'))
    await h.start(assistant(model, memory='stateful'))
    session = '00000000-0000-0000-0000-0000000000aa'
    assert (await h.ask('assistant', 'first', session_id=session)).output == 'a'
    assert (await h.ask('assistant', 'second', session_id=session)).output == 'b'
    assert [entry[1] for entry in seen] == [0, 0]  # every consult is a new peer session
    assert seen[0][2] != seen[1][2] and session not in {seen[0][2], seen[1][2]}


async def test_the_model_can_consult_only_allowed_peers(runtime_harness):
    from agentweave import Peers
    from agentweave.testing import FakeModel, text_turn
    h, seen, other = runtime_harness, [], []
    model = FakeModel([consult_turn('other', 'sneaky'), text_turn('refused')])
    await h.start(gated_analyst(None, seen))
    await h.start(gated_analyst(None, other, name='other'))
    await h.start(assistant(model, peers=Peers.allow(['analyst'])))
    assert (await h.ask('assistant', 'hi')).output == 'refused'
    assert model.requests[0].tools[0].input_schema['properties']['agent']['enum'] == ['analyst']
    assert model.requests[1].messages[-1].tool_results[0]['status'] == 'invalid'
    assert other == []


async def test_consulting_an_ancestor_is_a_cycle(runtime_harness):
    from agentweave.testing import FakeModel, text_turn
    h = runtime_harness
    parent = FakeModel([consult_turn('analyst', 'help'), text_turn('done')])
    child = FakeModel([consult_turn('assistant', 'back to you'), text_turn('I answer myself')])
    await h.start(assistant(parent))
    await h.start(assistant(child, name='analyst'))
    result = await h.ask('assistant', 'go')
    assert (result.status, result.output) == ('succeeded', 'done')
    refused = child.requests[1].messages[-1].tool_results[0]
    assert refused['status'] == 'invalid' and refused['error']['type'] == 'CycleDetected'
    assert len(parent.requests) == 2


async def test_a_waiting_parent_recovers_after_its_worker_restarts(runtime_harness):
    from agentweave.testing import FakeModel, text_turn
    h, seen, gate = runtime_harness, [], asyncio.Event()
    model = FakeModel([consult_turn('analyst', 'x'), text_turn('recovered')])
    await h.start(gated_analyst(gate, seen))
    await h.start(assistant(model))
    first = await h.client.agent('assistant').submit('go')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    await h.restart('assistant')
    gate.set()
    result = await asyncio.wait_for(h.client.result(first.task_id), 20)
    assert (result.status, result.output) == ('succeeded', 'recovered')
    assert len(model.requests) == 2 and len(seen) == 1  # step 0 replayed, the child asked once


async def test_a_python_handler_waits_for_a_consult_without_holding_its_slot(runtime_harness):
    from agentweave import Agent, Peers
    h, seen, gate = runtime_harness, [], asyncio.Event()
    helper = Agent('helper', peers=Peers.discover(), concurrency=1)

    @helper.on_request
    async def ask_analyst(ctx, request):
        if request.input == 'quick':
            return 'quick'
        result = await ctx.consult('analyst', request.input)
        return result.output['answer']
    await h.start(gated_analyst(gate, seen))
    await h.start(helper)
    first = await h.client.agent('helper').submit('deep')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    assert (await h.ask('helper', 'quick')).output == 'quick'
    gate.set()
    result = await asyncio.wait_for(h.client.result(first.task_id), 20)
    assert (result.status, result.output) == ('succeeded', 'deep')


async def test_a_python_consult_to_an_unallowed_peer_is_refused_before_sending(runtime_harness):
    from agentweave import Agent, Peers
    from agentweave.errors import ConsultRefused
    h, other, outcomes = runtime_harness, [], []
    helper = Agent('helper', peers=Peers.allow(['analyst']))

    @helper.on_request
    async def ask(ctx, request):
        try:
            await ctx.consult('other', 'x')
        except ConsultRefused as error:
            outcomes.append(type(error).__name__)
        return 'ok'
    await h.start(gated_analyst(None, other, name='other'))
    await h.start(helper)
    assert (await h.ask('helper', 'go')).output == 'ok'
    assert outcomes == ['ConsultRefused'] and other == []


async def test_concurrent_children_share_the_root_token_budget(runtime_harness):
    from agentweave.models import ModelBudget
    from agentweave.testing import FakeModel, text_turn
    h = runtime_harness
    half = {'input_tokens': 25, 'output_tokens': 25}
    two = consult_turn('analyst', 'a', call_id='c1')
    two = two.model_copy(update={'tool_calls': [two.tool_calls[0], consult_turn('critic', 'b', call_id='c2').tool_calls[0]]})
    parent = FakeModel([two, text_turn('never sent')])
    analyst, critic = FakeModel([text_turn('A', usage=half)]), FakeModel([text_turn('B', usage=half)])
    await h.start(assistant(analyst, name='analyst'))
    await h.start(assistant(critic, name='critic'))
    await h.start(assistant(parent, budget=ModelBudget(max_tokens=100)))
    result = await h.ask('assistant', 'go')
    assert result.status == 'failed' and result.error['type'] == 'ModelBudgetExceeded'
    assert len(parent.requests) == 1  # 50 + 50 used the whole root budget: no next request
    assert len(analyst.requests) == len(critic.requests) == 1


async def test_a_late_reply_does_not_continue_an_expired_parent(runtime_harness, fake_clock):
    from agentweave.testing import FakeModel, text_turn
    h, seen, gate = runtime_harness, [], asyncio.Event()
    model = FakeModel([consult_turn('analyst', 'x'), text_turn('too late')])
    await h.start(gated_analyst(gate, seen))
    await h.start(assistant(model))
    first = await h.client.agent('assistant').submit('go')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    fake_clock.advance(121)
    gate.set()
    await eventually(lambda: _status(h, first.task_id, 'expired'))
    await asyncio.sleep(.3)
    assert len(model.requests) == 1


import pytest  # noqa: E402


@pytest.mark.parametrize('safe', [True, False])
async def test_a_handler_that_dies_awaiting_a_consult_reruns_only_when_retry_safe(runtime_harness, safe):
    from agentweave import Agent, Peers
    h, seen, gate, runs = runtime_harness, [], asyncio.Event(), []
    helper = Agent('helper', peers=Peers.discover())

    @helper.on_request(retry='safe' if safe else None)
    async def ask(ctx, request):
        runs.append(request.input)
        result = await ctx.consult('analyst', request.input)
        return result.output['answer']
    await h.start(gated_analyst(gate, seen))
    await h.start(helper)
    first = await h.client.agent('helper').submit('deep')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    await h.workers['helper'].stop()  # the process dies while the handler waits
    await h.runs['helper']
    with sqlite3.connect(h.store.path, timeout=10) as connection:
        connection.execute("UPDATE tasks SET lease_expires_at='2000-01-01T00:00:00.000000+00:00' WHERE task_id=?",
                           (first.task_id,))
    await h.start(helper)
    gate.set()
    result = await asyncio.wait_for(h.client.result(first.task_id), 20)
    if safe:
        assert (result.status, result.output) == ('succeeded', 'deep')
        assert runs == ['deep', 'deep'] and len(seen) == 1  # the rerun found the same child, asked once
    else:
        assert result.status == 'outcome_unknown' and runs == ['deep']


async def test_a_waiting_parent_expires_at_its_deadline_even_without_any_reply(runtime_harness, fake_clock):
    from agentweave.testing import FakeModel, text_turn
    h, seen, gate = runtime_harness, [], asyncio.Event()  # the child never answers
    model = FakeModel([consult_turn('analyst', 'x'), text_turn('never')])
    await h.start(gated_analyst(gate, seen))
    await h.start(assistant(model))
    first = await h.client.agent('assistant').submit('go')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    fake_clock.advance(121)
    await eventually(lambda: _status(h, first.task_id, 'expired'))
    gate.set()


async def test_waiting_parents_do_not_use_up_admission_capacity(runtime_harness):
    from agentweave.testing import FakeModel, text_turn
    h, seen, gate = runtime_harness, [], asyncio.Event()
    model = FakeModel([consult_turn('analyst', 'slow'), text_turn('second'), text_turn('first done')])
    await h.start(gated_analyst(gate, seen))
    await h.start(assistant(model, concurrency=1, max_inflight=1))
    first = await h.client.agent('assistant').submit('first')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    assert (await h.ask('assistant', 'second')).output == 'second'
    gate.set()
    assert (await asyncio.wait_for(h.client.result(first.task_id), 20)).output == 'first done'


async def test_concurrent_consults_of_one_handler_keep_its_slot_accounting(runtime_harness):
    from agentweave import Agent, Peers
    h, seen = runtime_harness, []
    helper = Agent('helper', peers=Peers.discover(), concurrency=1)

    @helper.on_request
    async def fan_out(ctx, request):
        if request.input == 'quick':
            return 'quick'
        answers = await asyncio.gather(ctx.consult('analyst', 'a'), ctx.consult('analyst', 'b'))
        return sorted(answer.output['answer'] for answer in answers)
    await h.start(gated_analyst(None, seen))
    await h.start(helper)
    assert (await h.ask('helper', 'both')).output == ['a', 'b']
    assert (await h.ask('helper', 'quick')).output == 'quick'  # no slot was lost


async def test_waiting_for_a_consult_does_not_count_against_the_handler_time_limit(runtime_harness):
    from agentweave import Agent, Peers
    from agentweave.definitions import RetryPolicy
    h, seen, gate = runtime_harness, [], asyncio.Event()
    helper = Agent('helper', peers=Peers.discover())

    @helper.on_request(retry=RetryPolicy(timeout=0.5))
    async def ask(ctx, request):
        result = await ctx.consult('analyst', request.input)
        return result.output['answer']
    await h.start(gated_analyst(gate, seen))
    await h.start(helper)
    first = await h.client.agent('helper').submit('slow')
    await eventually(lambda: _status(h, first.task_id, 'waiting_peer'))
    await asyncio.sleep(1)  # longer than the handler's own limit
    gate.set()
    result = await asyncio.wait_for(h.client.result(first.task_id), 20)
    assert (result.status, result.output) == ('succeeded', 'slow')
