"""Handing off tasks and conversations: one owner at a time, atomic transfer, shared limits."""
import asyncio
import sqlite3
import pytest
from tests.integration.test_python_runtime import eventually, runtime_harness  # noqa: F401


def rows(store, sql, values=()):
    with sqlite3.connect(store.path, timeout=10) as connection:
        return connection.execute(sql, values).fetchall()


def passer(calls, target='analyst', name='general', scope=None, gate=None, message='over to you'):
    from agentweave import Agent, Peers
    agent = Agent(name, peers=Peers.discover())

    @agent.on_request
    async def hand_off(ctx, request):
        calls.append(request.input)
        if gate is not None:
            await gate.wait()
        ctx.handoff(target, message=message, scope=scope)
        calls.append('after handoff')  # never runs
    return agent


def answerer(seen, name='analyst', **options):
    from agentweave import Agent
    agent = Agent(name, **options)

    @agent.on_request
    async def answer(ctx, request):
        seen.append((request.input, [message.content for message in request.history]))
        return {'agent': name, 'input': request.input}
    return agent


async def _is(value, expected):
    return value == expected


async def test_next_message_uses_handoff_owner(runtime_harness):
    h, general, seen = runtime_harness, [], []
    await h.start(passer(general))
    await h.start(answerer(seen))
    conversation = await h.client.conversation('general')
    first = await asyncio.wait_for(conversation.ask('first question'), 20)
    assert (first.status, first.responding_agent) == ('succeeded', 'analyst')
    assert first.output == {'agent': 'analyst', 'input': 'first question'}
    assert 'over to you' in seen[0][1]  # the handoff message travels with the task
    assert await conversation.refresh() == 'analyst'
    second = await asyncio.wait_for(conversation.ask('ещё вопрос'), 20)
    assert second.responding_agent == 'analyst' and seen[-1][0] == 'ещё вопрос'
    assert general == ['first question']  # the old owner is never asked again


async def test_a_task_only_handoff_keeps_the_conversation_owner(runtime_harness):
    h, general, seen = runtime_harness, [], []
    await h.start(passer(general, scope='task'))
    await h.start(answerer(seen))
    conversation = await h.client.conversation('general')
    assert (await asyncio.wait_for(conversation.ask('one'), 20)).responding_agent == 'analyst'
    assert await conversation.refresh() == 'general'
    await asyncio.wait_for(conversation.ask('two'), 20)
    assert general == ['one', 'two']


async def test_a_one_shot_request_hands_off_only_its_task(runtime_harness):
    h, general, seen = runtime_harness, [], []
    await h.start(passer(general))
    await h.start(answerer(seen))
    assert (await h.ask('general', 'a')).responding_agent == 'analyst'
    assert (await h.ask('general', 'b')).responding_agent == 'analyst'
    assert general == ['a', 'b']  # each independent request still goes to general first


async def test_the_recipient_sees_only_this_conversation_never_its_other_sessions(runtime_harness):
    h, general, seen = runtime_harness, [], []
    await h.start(passer(general))
    await h.start(answerer(seen, memory='stateful'))
    await h.ask('analyst', 'private earlier question', session_id='00000000-0000-0000-0000-0000000000bb')
    conversation = await h.client.conversation('general')
    await asyncio.wait_for(conversation.ask('hello'), 20)
    history = seen[-1][1]
    assert 'private earlier question' not in history and 'over to you' in history


async def test_a_message_queued_during_a_handoff_goes_to_the_new_owner(runtime_harness):
    h, general, seen, gate = runtime_harness, [], [], asyncio.Event()
    await h.start(passer(general, gate=gate))
    await h.start(answerer(seen))
    conversation = await h.client.conversation('general')
    first = await conversation.submit('first')
    second = await conversation.submit('second')  # sent while general still owns the conversation
    await eventually(lambda: _is(general, ['first']))
    gate.set()
    for submission in (first, second):
        assert (await asyncio.wait_for(h.client.result(submission.task_id), 20)).responding_agent == 'analyst'
    assert general == ['first'] and [entry[0] for entry in seen] == ['first', 'second']


async def test_the_old_owner_cannot_commit_after_the_transfer(store, envelope_factory):
    from agentweave._records import StepCommit
    from agentweave.errors import LeaseLost
    event = envelope_factory(recipient='general')
    await store.admit(event, 'def-general')
    await store.register_definition('analyst', '00000000-0000-0000-0000-0000000000c2', 'def-analyst')
    lease = await store.claim(event.task_id, '00000000-0000-0000-0000-0000000000c1', 30)
    task = await store.get_task(event.task_id)
    moved = await store.transfer(lease, 'analyst', {'messages': []}, 'task')
    assert (moved.task_id, moved.agent_id, moved.status) == (event.task_id, 'analyst', 'accepted')
    with pytest.raises(LeaseLost):
        await store.commit_step(lease, StepCommit(step_id='late', expected_version=task.version, status='succeeded'))


async def test_a_repeated_handoff_notice_runs_the_new_owner_once(runtime_harness):
    from agentweave.messages import decode
    h, general, seen = runtime_harness, [], []
    await h.start(passer(general))
    await h.start(answerer(seen))
    result = await h.ask('general', 'x')
    [(raw,)] = rows(h.store, "SELECT envelope FROM outbox WHERE json_extract(envelope,'$.kind')='handoff'")
    for _ in range(2):
        await h.bus.publish(decode(raw.encode()))
    await asyncio.sleep(.5)
    assert result.responding_agent == 'analyst' and len(seen) == 1


async def test_a_managed_handoff_is_a_lone_control_call_with_portable_context(runtime_harness):
    from agentweave import Agent, Model, Peers, tool
    from agentweave.messages import Message
    from agentweave.models import ModelTurn, ProposedToolCall
    from agentweave.testing import FakeModel, tool_turn
    h, seen, effects = runtime_harness, [], []

    @tool
    def note(text: str) -> str:
        """Write a note."""
        effects.append(text)
        return 'noted'
    both = ModelTurn(messages=[Message(role='assistant', content='thinking', adapter_metadata={'opaque': 'secret'})],
                     tool_calls=[ProposedToolCall(provider_call_id='c1', name='handoff_to',
                                                  arguments={'agent': 'analyst', 'message': 'yours'}),
                                 ProposedToolCall(provider_call_id='c2', name='note', arguments={'text': 'x'})],
                     usage={'input_tokens': 1, 'output_tokens': 1})
    model = FakeModel([both, tool_turn('handoff_to', {'agent': 'analyst', 'message': 'yours'}, call_id='c3')])
    general = Agent('general', instructions='SYSTEM SECRET', model=Model.custom(model), tools=[note],
                    peers=Peers.discover())
    await h.start(general)
    await h.start(answerer(seen))
    result = await h.ask('general', 'help me')
    assert result.responding_agent == 'analyst' and effects == []  # the mixed turn ran nothing
    assert all(r['status'] in ('invalid', 'skipped') for r in model.requests[1].messages[-1].tool_results)
    context = seen[0][1]
    assert 'yours' in context and 'thinking' in context
    assert 'SYSTEM SECRET' not in str(context) and "'opaque'" not in str(context)


async def test_a_handoff_inherits_the_spent_root_budget(runtime_harness):
    from agentweave import Agent, Model, Peers
    from agentweave.models import ModelBudget
    from agentweave.testing import FakeModel, text_turn, tool_turn
    h = runtime_harness
    spent = {'input_tokens': 50, 'output_tokens': 50}
    general = FakeModel([tool_turn('handoff_to', {'agent': 'analyst', 'message': 'yours'}, usage=spent)])
    analyst = FakeModel([text_turn('never sent')])
    await h.start(Agent('general', model=Model.custom(general), peers=Peers.discover(),
                        budget=ModelBudget(max_tokens=100)))
    await h.start(Agent('analyst', model=Model.custom(analyst)))
    result = await h.ask('general', 'go')
    assert result.responding_agent == 'analyst'
    assert result.status == 'failed' and result.error['type'] == 'ModelBudgetExceeded'
    assert analyst.requests == []  # no fresh allowance for the recipient


async def test_at_most_eight_handoffs_per_task(runtime_harness):
    h, ping, pong = runtime_harness, [], []
    await h.start(passer(ping, target='pong', name='ping'))
    await h.start(passer(pong, target='ping', name='pong'))
    result = await h.ask('ping', 'loop')
    assert result.status == 'failed' and result.error['type'] == 'HandoffRefused'
    assert len(ping) + len(pong) == 9  # the ninth owner may not pass it on
    assert rows(h.store, "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='handoff'",
                (result.task_id,)) == [(8,)]


async def test_a_recipient_must_be_able_to_take_the_task(runtime_harness):
    from agentweave import Agent
    h, general = runtime_harness, []
    listener = Agent('listener')

    @listener.on('news.posted')
    async def read(ctx, event):
        return None
    await h.start(passer(general, target='listener'))
    await h.start(listener)
    result = await h.ask('general', 'x')
    assert result.status == 'failed' and result.error['type'] == 'HandoffRefused'
    assert result.responding_agent == 'general'


async def test_an_oversized_context_is_refused_before_the_owner_changes(runtime_harness):
    h, general, seen = runtime_harness, [], []
    await h.start(passer(general, message='x' * 1_100_000))
    await h.start(answerer(seen))
    result = await h.ask('general', 'x')
    assert result.status == 'failed' and result.error['type'] == 'HandoffRefused'
    assert result.responding_agent == 'general' and seen == []


async def test_a_stateful_recipient_gets_the_senders_turns_once_and_only_portably(runtime_harness):
    from agentweave import Agent, Peers
    h, seen = runtime_harness, []
    general = Agent('general', memory='stateful', peers=Peers.discover())

    @general.on_request
    async def answer_then_pass(ctx, request):
        if request.input == 'turn 1':
            return 'answer 1'
        ctx.handoff('analyst', message='over to you')
    await h.start(general)
    await h.start(answerer(seen, memory='stateful'))
    conversation = await h.client.conversation('general')
    await asyncio.wait_for(conversation.ask('turn 1'), 20)
    await asyncio.wait_for(conversation.ask('turn 2'), 20)
    assert seen[0][1] == ['turn 1', 'answer 1', 'over to you']  # no copy of general's raw session


async def test_a_task_only_helper_cannot_take_over_the_conversation(runtime_harness):
    h, general, helper, seen = runtime_harness, [], [], []
    await h.start(passer(general, target='helper', scope='task'))
    await h.start(passer(helper, target='analyst', name='helper'))  # no explicit scope
    await h.start(answerer(seen))
    conversation = await h.client.conversation('general')
    assert (await asyncio.wait_for(conversation.ask('x'), 20)).responding_agent == 'analyst'
    assert await conversation.refresh() == 'general'


async def test_a_message_its_conversation_owner_cannot_take_is_answered_where_it_arrived(runtime_harness):
    h, seen = runtime_harness, []
    await h.start(answerer(seen, name='general'))
    conversation = await h.client.conversation('general')
    with sqlite3.connect(h.store.path, timeout=10) as connection:
        connection.execute("UPDATE conversations SET active_agent='ghost' WHERE session_id=?",
                           (conversation.conversation_id,))
    submission = await h.client.agent('general').submit('hello', conversation.conversation_id)
    result = await asyncio.wait_for(h.client.result(submission.task_id), 20)
    assert (result.status, result.responding_agent) == ('succeeded', 'general')


async def test_reattaching_to_an_unknown_conversation_fails_clearly(runtime_harness):
    from agentweave.errors import ConversationNotFound
    h, seen = runtime_harness, []
    await h.start(answerer(seen, name='general'))
    with pytest.raises(ConversationNotFound):
        await h.client.conversation('general', '00000000-0000-0000-0000-0000000000ee')
