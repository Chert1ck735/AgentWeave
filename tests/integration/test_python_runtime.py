"""Real journal and broker runtime contracts, with bounded deterministic checks."""
import asyncio
from datetime import timedelta
from uuid import UUID, uuid4
import pytest

OWNER = '00000000-0000-0000-0000-000000000001'
OTHER = '00000000-0000-0000-0000-000000000002'

async def eventually(check, timeout=20):
    async with asyncio.timeout(timeout):
        while True:
            value = await check()
            if value:
                return value
            await asyncio.sleep(.02)

async def test_workload_duplicate_capacity_watermark_and_restart(store, store_factory, envelope_factory):
    from agentweave._records import EventRejection
    snap = envelope_factory(kind='numbers.received', recipient=None, topic='numbers.received',
        key='counter', source_version=3, task_id=None)
    first = await store.admit_with_policy(snap, 'v1', agent_id='counter', max_inflight=1, snapshot=True)
    assert not first.duplicate
    duplicate = await store.admit_with_policy(snap, 'changed', agent_id='counter', max_inflight=1, snapshot=True)
    assert duplicate.duplicate and duplicate.receipt == first.receipt
    old = snap.model_copy(update={'event_id':str(uuid4()), 'source_version':2})
    rejection = await store.admit_with_policy(old, 'v1', agent_id='counter', max_inflight=1, snapshot=True)
    assert rejection == EventRejection(reason='stale_version', retryable=False)
    newer = snap.model_copy(update={'event_id':str(uuid4()), 'source_version':4})
    assert (await store.admit_with_policy(newer, 'v1', agent_id='counter', max_inflight=1, snapshot=True)).retryable
    rival = await store_factory(store.path)
    assert (await rival.admit_with_policy(old, 'v1', agent_id='counter', max_inflight=64, snapshot=True)).reason == 'stale_version'
    assert (await rival.get_task(first.task_id)).status == 'accepted'

async def test_atomic_replica_capacity(store, store_factory, envelope_factory):
    from agentweave._records import Admission, EventRejection
    rival = await store_factory(store.path)
    results = await asyncio.gather(*[s.admit_with_policy(envelope_factory(), 'v1', agent_id='assistant', max_inflight=1) for s in (store, rival)])
    assert sum(isinstance(x, Admission) for x in results) == 1
    assert sum(isinstance(x, EventRejection) and x.retryable for x in results) == 1

async def test_recovery_pages_progress_and_gates(store, envelope_factory):
    from agentweave._records import StepCommit
    events = [envelope_factory(session_id=OWNER) for _ in range(67)]
    for event in events:
        await store.admit_with_policy(event, 'v1', agent_id='assistant', max_inflight=100)
    page = await store.recoverable_page('assistant', limit=64)
    tail = await store.recoverable_page('assistant', after_task_id=page[-1].task_id, limit=64)
    assert len(page) == 64 and len(tail) == 3
    assert len({x.task_id for x in page + tail}) == 67
    first, second = events[:2]
    left = await store.claim(first.task_id, OWNER, 30)
    right = await store.claim(second.task_id, OTHER, 30)
    assert await store.acquire_key_gate(first.task_id, '$session', OWNER, lease=left)
    assert not await store.acquire_key_gate(second.task_id, '$session', OWNER, lease=right)
    progress = first.model_copy(update={'event_id':str(uuid4()), 'kind':'progress', 'recipient':None, 'topic':'progress'})
    sequence = await store.append_progress(first.task_id, progress)
    assert await store.append_progress(first.task_id, progress) == sequence
    assert await store.read_progress(first.task_id) == [progress]
    assert await store.read_progress(first.task_id, after_sequence=sequence) == []
    task = await store.get_task(first.task_id)
    await store.commit_step(left, StepCommit(step_id='done', expected_version=task.version, status='succeeded'))
    assert await store.acquire_key_gate(second.task_id, '$session', OWNER, lease=right)

async def test_expiry_before_start_and_unknown_started_handler(store, envelope_factory, fake_clock):
    soon = envelope_factory(expires_at=fake_clock.now() + timedelta(seconds=1))
    await store.admit_with_policy(soon, 'v1', agent_id='assistant', max_inflight=64)
    fake_clock.advance(2)
    assert (await store.expire_task(soon.task_id)).status == 'expired'
    running = envelope_factory()
    await store.admit_with_policy(running, 'v1', agent_id='assistant', max_inflight=64)
    lease = await store.claim(running.task_id, OWNER, 1)
    assert await store.start_handler(lease, retry_safe=False)
    fake_clock.advance(121)
    assert (await store.expire_task(running.task_id)).status == 'outcome_unknown'

async def test_client_connect_prepares_first_run_and_closes_only_owned_adapters(tmp_path):
    from agentweave import Client
    from agentweave._infrastructure.processes import ProcessRegistry
    from agentweave.cli.infra import down
    config = None
    try:
        async with Client.connect(tmp_path) as client:
            config = client.config
            assert client.store.path == tmp_path/'.agentweave/dev/state.sqlite3'
            assert config.broker_url.startswith('127.0.0.1:')
            assert config.broker_epoch
            assert client.directory.snapshot() == {}
        registry = ProcessRegistry(tmp_path)
        broker = next(r for r in registry.records() if r.logical_id == '$broker')
        assert registry.verify(broker)
        async with Client.connect(tmp_path) as again:
            assert again.config.broker_epoch == config.broker_epoch
            assert again.config.network_id == config.network_id
    finally:
        if config is not None:
            await asyncio.to_thread(down,tmp_path,'dev')
    assert (tmp_path/'.agentweave/dev/state.sqlite3').is_file()


class RuntimeHarness:
    def __init__(self, config, store, bus, directory):
        from agentweave import Client
        self.config, self.store, self.bus, self.directory = config, store, bus, directory
        self.client = Client(config, store, bus, directory)
        self.workers, self.runs, self.definitions = {}, {}, {}

    async def start(self, definition):
        from agentweave import Worker
        worker = Worker(definition, self.config, self.store, self.bus, self.directory)
        key = definition.name
        self.workers[key], self.definitions[key] = worker, definition
        self.runs[key] = asyncio.create_task(worker.run())
        await asyncio.wait_for(worker.ready.wait(), 20)
        async def announced():
            return bool(self.directory.snapshot().get(key))
        await eventually(announced)
        return worker

    async def ask(self, name, input, session_id=None):
        return await asyncio.wait_for(self.client.agent(name).ask(input, session_id), 20)

    async def restart(self, name):
        await self.workers[name].stop()
        await self.runs[name]
        return await self.start(self.definitions[name])

    async def close(self):
        for worker in self.workers.values():
            await worker.stop()
        await asyncio.gather(*self.runs.values())
        await self.client.close()


@pytest.fixture
async def runtime_harness(broker_process, store, tmp_path, fake_clock):
    from agentweave.config import ProfileConfig
    from agentweave.adapters.broker import BrokerBus
    from agentweave._directory import LiveDirectory
    config = ProfileConfig(profile='dev', root=tmp_path, project_id=str(uuid4()),
        network_id='00000000-0000-0000-0000-000000000010', broker_url=broker_process.url,
        database_url='sqlite:///'+str(store.path), broker_epoch=str(UUID(broker_process.epoch)))
    bus = BrokerBus(config.broker_url, config.network_id, profile='dev', broker_epoch=config.broker_epoch)
    directory = LiveDirectory(bus, store, config.network_id, broker_epoch=config.broker_epoch, clock=fake_clock)
    harness = RuntimeHarness(config, store, bus, directory)
    try:
        yield harness
    finally:
        await harness.close()
        await directory.close()
        await bus.close()


async def test_state_survives_restart_and_event_path(runtime_harness, envelope_factory):
    from tests.fixtures.agents import counter_agent
    h = runtime_harness
    await h.start(counter_agent())
    assert (await h.ask('counter', [1], session_id=OWNER)).output == 1
    await h.restart('counter')
    assert (await h.ask('counter', [2], session_id=OWNER)).output == 2
    event = envelope_factory(kind='numbers.received', topic='numbers.received', recipient=None,
        task_id=None, session_id=OWNER, payload={'numbers':[3,4]})
    await h.bus.publish(event)
    from uuid import UUID, uuid5
    task_id = str(uuid5(UUID(event.event_id), 'task:counter'))
    async def event_admitted():
        return await h.store.get_task(task_id)
    await eventually(event_admitted)
    result = await asyncio.wait_for(h.client.result(task_id),20)
    assert result.output == 4
    state, history = await h.store.load_session('counter', OWNER)
    assert state == {'count':4} and len(history) == 6


async def test_stateless_duplicate_and_client_close_do_not_cancel(runtime_harness):
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    await h.start(echo_agent())
    first = await h.client.agent('echo').submit('one', session_id=OWNER)
    result = await asyncio.wait_for(h.client.result(first.task_id),20)
    assert result.output == {'input':'one','history':0}
    event = await h.store.get_input_envelope(first.task_id)
    await h.bus.publish(event)
    assert (await h.ask('echo','two',session_id=OWNER)).output['history'] == 0
    assert (await h.store.load_session('echo',OWNER)) == ({},[])
    pending = await h.client.agent('echo').submit('close')
    await h.client.close()
    async def complete():
        task = await h.store.get_task(pending.task_id)
        return task if task and task.status == 'succeeded' else None
    assert (await eventually(complete)).result['input'] == 'close'


async def test_buffered_publish_atomic_and_failed_state_not_saved(runtime_harness):
    from agentweave import Agent
    h = runtime_harness
    bad = Agent('bad',memory='stateful')
    @bad.on_request
    async def fail(ctx, request):
        ctx.state['dirty'] = True
        await ctx.publish('should.not.send', {'secret':'never'})
        raise ValueError('not recorded secret')
    await h.start(bad)
    result = await h.ask('bad','value',session_id=OWNER)
    assert result.status == 'failed'
    assert await h.store.load_session('bad',OWNER) == ({},[])
    import sqlite3
    with sqlite3.connect(h.store.path) as c:
        assert c.execute("SELECT COUNT(*) FROM outbox WHERE json_extract(envelope,'$.topic')='should.not.send'").fetchone()[0] == 0
    from tests.fixtures.agents import counter_agent
    await h.start(counter_agent())
    completed = await h.ask('counter',[1],session_id=OTHER)
    with sqlite3.connect(h.store.path) as c:
        rows = c.execute("SELECT json_extract(envelope,'$.payload.count') FROM outbox WHERE json_extract(envelope,'$.topic')='count.updated'").fetchall()
        assert rows == [(1,)]
        assert c.execute("SELECT status FROM tasks WHERE task_id=?",(completed.task_id,)).fetchone() == ('succeeded',)


async def test_current_directory_capability_and_stateful_session_rejection(runtime_harness, envelope_factory):
    from tests.fixtures.agents import counter_agent
    from agentweave.errors import DefinitionError
    h = runtime_harness
    w = await h.start(counter_agent())
    with pytest.raises(DefinitionError):
        await h.client.agent('missing').submit('x')
    event = envelope_factory(kind='numbers.received',topic='numbers.received',recipient=None,
        session_id=None,task_id=None,payload={'numbers':[1]})
    await h.bus.publish(event)
    async def rejected():
        return w.metrics.counters['rejected']
    await eventually(rejected)
    assert await h.store.recoverable_page('counter') == []


async def test_gates_fence_stale_owner_and_retry_attempt_budget(store, envelope_factory, fake_clock):
    from agentweave.errors import LeaseLost
    event = envelope_factory()
    await store.admit_with_policy(event,'v1',agent_id='assistant',max_inflight=64)
    old = await store.claim_runtime(event.task_id, OWNER, 1, concurrency=1)
    assert await store.acquire_key_gate(event.task_id,'request',event.session_id,lease=old)
    assert await store.start_handler(old,retry_safe=True,max_attempts=2)
    fake_clock.advance(2)
    new = await store.claim_runtime(event.task_id, OTHER, 30, concurrency=1)
    with pytest.raises(LeaseLost):
        await store.acquire_key_gate(event.task_id,'request',event.session_id,lease=old)
    assert await store.acquire_key_gate(event.task_id,'request',event.session_id,lease=new)
    assert await store.start_handler(new,retry_safe=True,max_attempts=2)
    await store.release_claim(new, error={'type':'ValueError','message':'failed'})
    final = await store.claim_runtime(event.task_id, OWNER,30,concurrency=1)
    assert not await store.start_handler(final,retry_safe=True,max_attempts=2)
    # The last attempt reported its failure, so exhausting the budget is a clean failure;
    # an attempt released without a recorded outcome would be outcome_unknown instead.
    assert (await store.get_task(event.task_id)).status == 'failed'


async def test_sync_timeout_keeps_executor_threads_bounded(runtime_harness):
    import threading
    from agentweave import Agent
    from agentweave.definitions import RetryPolicy
    h = runtime_harness
    agent = Agent('sync', concurrency=1)
    release = threading.Event()
    entered = []
    @agent.on_request(retry=RetryPolicy(timeout=.05))
    def blocked(ctx, request):
        entered.append(request.input)
        release.wait(5)
        ctx.state['late'] = True
        return 'late'
    await h.start(agent)
    try:
        first = await h.ask('sync','first')
        first_id = first.task_id
        assert first.status == 'outcome_unknown'
        second = await h.client.agent('sync').submit('second')
        async def admitted_second():
            return await h.store.get_task(second.task_id)
        task = await eventually(admitted_second)
        await asyncio.sleep(.1)
        assert entered == ['first']
        assert task.result is None
    finally:
        release.set()
    assert (await h.store.get_task(first_id)).result is None


async def test_unknown_retains_gate_and_logical_execution_limit(store, envelope_factory):
    from agentweave._records import StepCommit
    first, second = [envelope_factory(session_id=OWNER) for _ in range(2)]
    for event in (first, second):
        await store.admit_with_policy(event,'v1',agent_id='assistant',max_inflight=64)
    lease = await store.claim_runtime(first.task_id,OWNER,30,concurrency=1)
    assert await store.claim_runtime(second.task_id,OTHER,30,concurrency=1) is None
    assert await store.acquire_key_gate(first.task_id,'$session',OWNER,lease=lease)
    task = await store.get_task(first.task_id)
    await store.commit_step(lease,StepCommit(step_id='unknown',expected_version=task.version,status='outcome_unknown'))
    rival = await store.claim_runtime(second.task_id,OTHER,30,concurrency=1)
    assert not await store.acquire_key_gate(second.task_id,'$session',OWNER,lease=rival)


async def test_pending_snapshot_queue_coalesces_versions_and_never_commands(fake_clock, envelope_factory):
    from agentweave import SubscriptionPolicy
    from agentweave._runtime.handlers import SubscriptionQueue
    from agentweave._runtime.logging import RuntimeMetrics
    from agentweave.ports import Delivery
    policy = SubscriptionPolicy(queue_limit=2,overflow='latest',replaceable=True)
    q = SubscriptionQueue(policy,fake_clock,RuntimeMetrics(),'numbers.received')
    three = envelope_factory(kind='numbers.received',recipient=None,topic='numbers.received',key='k',source_version=3)
    two = three.model_copy(update={'event_id':str(uuid4()),'source_version':2})
    four = three.model_copy(update={'event_id':str(uuid4()),'source_version':4})
    assert await q.put(Delivery(three,1)) == (None,None)
    assert (await q.put(Delivery(two,2)))[1] == 'stale_version'
    dropped, reason = await q.put(Delivery(four,3))
    assert reason == 'coalesced' and dropped.envelope == three
    command = three.model_copy(update={'event_id':str(uuid4()),'recipient':'counter','topic':None,'kind':'request'})
    assert await q.put(Delivery(command,4)) == (None,None)
    delta = three.model_copy(update={'event_id':str(uuid4()),'source_version':None})
    assert (await q.put(Delivery(delta,5)))[1] == 'rejected'
    assert q.metrics.queues['numbers.received']['depth'] == 2
    assert (await q.get()).envelope == four
    assert (await q.get()).envelope == command


async def test_slow_subscription_does_not_block_fast_or_receipts(runtime_harness, envelope_factory):
    from agentweave import Agent, SubscriptionPolicy
    from uuid import uuid5
    h = runtime_harness
    blocked, release = asyncio.Event(), asyncio.Event()
    slow = Agent('slow',max_inflight=1)
    @slow.on('broadcast',policy=SubscriptionPolicy(queue_limit=1,overflow='wait'))
    async def stall(ctx,event):
        blocked.set()
        await release.wait()
        return event
    fast = Agent('fast')
    @fast.on('broadcast',policy=SubscriptionPolicy(queue_limit=4,overflow='wait'))
    async def quick(ctx,event):
        return event
    wslow = await h.start(slow)
    await h.start(fast)
    first = envelope_factory(kind='broadcast',recipient=None,topic='broadcast',task_id=None,payload=1)
    await h.bus.publish(first)
    await asyncio.wait_for(blocked.wait(),20)
    later = [first.model_copy(update={'event_id':str(uuid4()),'payload':x}) for x in (2,3,4)]
    for event in later:
        await h.bus.publish(event)
    task_id = str(uuid5(UUID(later[-1].event_id),'task:fast'))
    try:
        async def fast_admitted():
            return await h.store.get_task(task_id)
        await eventually(fast_admitted)
        assert (await asyncio.wait_for(h.client.result(task_id),20)).output == 4
        assert all(x['depth'] <= 1 for x in wslow.metrics.queues.values())
        assert wslow.pump.last_error is None
    finally:
        release.set()


async def test_drain_finishes_running_handler(runtime_harness):
    from agentweave import Agent
    h = runtime_harness
    entered, release = asyncio.Event(), asyncio.Event()
    agent = Agent('draining')
    @agent.on_request
    async def finish(ctx,request):
        entered.set()
        await release.wait()
        return 'finished'
    worker = await h.start(agent)
    submission = await h.client.agent('draining').submit('x')
    await asyncio.wait_for(entered.wait(),20)
    drain = asyncio.create_task(worker.drain())
    await asyncio.sleep(.1)
    assert not drain.done()
    release.set()
    await asyncio.wait_for(drain,20)
    assert (await h.store.get_task(submission.task_id)).status == 'succeeded'


async def test_progress_stream_reconnect_has_no_duplicates(runtime_harness):
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    await h.start(echo_agent())
    result = await h.ask('echo','progress')
    original = await h.store.get_input_envelope(result.task_id)
    progress = original.model_copy(update={'event_id':str(uuid4()),'kind':'progress',
        'recipient':None,'topic':'progress','payload':{'stage':1}})
    await h.store.append_progress(result.task_id,progress)
    await h.bus.publish(progress)
    assert [event async for event in h.client.events(result.task_id)] == [progress]
    assert [event async for event in h.client.events(result.task_id)] == [progress]


@pytest.mark.integration
@pytest.mark.parametrize('retry_safe',[False,True])
async def test_real_process_kill_rollover_safe_restart_or_unknown(runtime_harness,tmp_path,fake_clock,retry_safe):
    import multiprocessing
    from tests.fixtures.agents import run_worker_process
    from agentweave import Client
    h = runtime_harness
    await h.directory.start()
    context = multiprocessing.get_context('spawn')
    clock_path = tmp_path/'clock.txt'
    clock_path.write_text(fake_clock.now().isoformat())
    ready, stop = context.Queue(),context.Event()
    processes = []
    def launch():
        p = context.Process(target=run_worker_process,args=(h.config.model_dump_json(),str(h.store.path),str(clock_path),'crash',retry_safe,ready,stop))
        p.start()
        processes.append(p)
        return p
    process = launch()
    try:
        outcome = await asyncio.to_thread(ready.get,True,25)
        assert outcome[0] == 'ready', outcome
        async def visible():
            return bool(h.directory.snapshot().get('crash'))
        await eventually(visible)
        initial = await h.ask('crash',{},session_id=OWNER)
        assert initial.output == 1
        marker = tmp_path/'started.txt'
        submission = await h.client.agent('crash').submit({'marker':str(marker)},session_id=OWNER)
        async def started():
            return marker.exists()
        await eventually(started)
        process.kill()
        await asyncio.to_thread(process.join,5)
        fake_clock.advance(31)
        clock_path.write_text(fake_clock.now().isoformat())
        process = launch()
        outcome = await asyncio.to_thread(ready.get,True,25)
        assert outcome[0] == 'ready',outcome
        result = await asyncio.wait_for(h.client.result(submission.task_id),20)
        assert result.status == ('succeeded' if retry_safe else 'outcome_unknown')
        state,history = await h.store.load_session('crash',OWNER)
        assert state == {'count':2 if retry_safe else 1}
        assert len(history) == (4 if retry_safe else 2)
        assert result.output == (2 if retry_safe else None)
        import sqlite3
        with sqlite3.connect(h.store.path) as c:
            assert c.execute('SELECT attempts FROM handler_attempts WHERE task_id=?',(submission.task_id,)).fetchone() == ((2,) if retry_safe else (1,))
    finally:
        stop.set()
        for p in processes:
            await asyncio.to_thread(p.join,15)
            if p.is_alive():
                p.kill()
                await asyncio.to_thread(p.join,5)
        ready.close()


@pytest.mark.integration
async def test_two_worker_processes_serialize_session_and_logical_execution(runtime_harness,tmp_path,fake_clock):
    import multiprocessing
    import sqlite3
    from tests.fixtures.agents import run_worker_process
    h = runtime_harness
    await h.directory.start()
    context = multiprocessing.get_context('spawn')
    clock_path = tmp_path/'clock.txt'
    clock_path.write_text(fake_clock.now().isoformat())
    ready, stop = context.Queue(),context.Event()
    processes = [context.Process(target=run_worker_process,args=(h.config.model_dump_json(),str(h.store.path),str(clock_path),'counter',True,ready,stop)) for _ in range(2)]
    for p in processes:
        p.start()
    try:
        for _ in processes:
            outcome = await asyncio.to_thread(ready.get,True,25)
            assert outcome[0] == 'ready',outcome
        async def replicas():
            return len(h.directory.snapshot().get('counter',())) == 2
        await eventually(replicas)
        submissions = [await h.client.agent('counter').submit([n],session_id=OWNER) for n in range(4)]
        results = await asyncio.gather(*[asyncio.wait_for(h.client.result(x.task_id),20) for x in submissions])
        assert sorted(x.output for x in results) == [1,2,3,4]
        assert (await h.store.load_session('counter',OWNER))[0] == {'count':4}
        with sqlite3.connect(h.store.path) as c:
            assert c.execute('SELECT SUM(attempts) FROM handler_attempts').fetchone() == (4,)
            assert c.execute("SELECT COUNT(*) FROM tasks WHERE status='succeeded'").fetchone() == (4,)
    finally:
        stop.set()
        for p in processes:
            await asyncio.to_thread(p.join,15)
            if p.is_alive():
                p.kill()
                await asyncio.to_thread(p.join,5)
        ready.close()


async def test_changed_fingerprint_refuses_recovery(runtime_harness,envelope_factory):
    from tests.fixtures.agents import counter_agent
    from agentweave.errors import StoreConflict
    from agentweave import Worker
    h = runtime_harness
    event = envelope_factory(recipient='counter',payload=[1])
    await h.store.admit_with_policy(event,'different',agent_id='counter',max_inflight=64)
    worker = Worker(counter_agent(),h.config,h.store,h.bus,h.directory)
    with pytest.raises(StoreConflict,match='nonterminal'):
        await worker.run()
    assert (await h.store.get_task(event.task_id)).status == 'accepted'


async def test_drain_reports_unknown_instead_of_claiming_completion(runtime_harness):
    from agentweave import Agent
    from agentweave.definitions import RetryPolicy
    h = runtime_harness
    agent = Agent('uncertain')
    @agent.on_request(retry=RetryPolicy(timeout=.01))
    async def unresolved(ctx,request):
        await asyncio.Event().wait()
    worker = await h.start(agent)
    assert (await h.ask('uncertain','x')).status == 'outcome_unknown'
    with pytest.raises(RuntimeError,match='unresolved'):
        await asyncio.wait_for(worker.drain(),5)


async def test_recovery_runs_independent_sessions_up_to_concurrency(runtime_harness,envelope_factory):
    from agentweave import Agent,Worker
    h = runtime_harness
    second_started = asyncio.Event()
    agent = Agent('parallel',memory='stateful',concurrency=2)
    @agent.on_request
    async def parallel(ctx,request):
        if request.input == 1:
            await second_started.wait()
        else:
            second_started.set()
        return request.input
    temporary = Worker(agent,h.config,h.store,h.bus,h.directory)
    for number,task_id in [(1,OWNER),(2,OTHER)]:
        event = envelope_factory(recipient='parallel',payload=number,task_id=task_id,session_id=task_id)
        await h.store.admit_with_policy(event,temporary.fingerprint,agent_id='parallel',max_inflight=64)
    temporary.engine.executor.close()
    await h.start(agent)
    async with asyncio.timeout(3):
        results = await asyncio.gather(h.client.result(OWNER),h.client.result(OTHER))
    assert [x.output for x in results] == [1,2]


async def test_invalid_handler_result_fails_atomically(runtime_harness):
    from agentweave import Agent
    h = runtime_harness
    agent = Agent('invalid',memory='stateful')
    @agent.on_request
    async def invalid(ctx,request):
        ctx.state['partial'] = True
        return object()
    await h.start(agent)
    result = await h.ask('invalid','x',session_id=OWNER)
    assert result.status == 'failed'
    assert await h.store.load_session('invalid',OWNER) == ({},[])


async def test_safe_retry_reloads_committed_state_and_stops_at_persisted_budget(runtime_harness):
    from agentweave import Agent
    from agentweave.definitions import RetryPolicy
    h = runtime_harness
    agent = Agent('retrying',memory='stateful')
    attempts = []
    @agent.on_request(retry=RetryPolicy(safe=True,max_attempts=2,min_delay=.01,max_delay=.02,jitter=False))
    async def failing(ctx,request):
        attempts.append(dict(ctx.state))
        ctx.state['partial'] = True
        raise ValueError('failed')
    await h.start(agent)
    asking = asyncio.create_task(h.ask('retrying','x',session_id=OWNER))
    # Each retry time is journaled after the handler fails, and the exhausted budget is found
    # when the next attempt is due: keep the store clock moving until the task ends.
    async def finished():
        h.store.clock.advance(.02)
        return asking.done()
    await eventually(finished)
    result = await asking
    assert result.status == 'failed'
    assert attempts == [{},{}]
    assert (await h.store.load_session('retrying',OWNER))[0] == {}


async def test_submission_expiry_is_persisted_without_fabricated_admission(runtime_harness,envelope_factory,fake_clock,store_factory):
    from agentweave.messages import TaskResult
    h = runtime_harness
    event = envelope_factory(deadline=fake_clock.now()+timedelta(seconds=1))
    await h.store.submit(event)
    fake_clock.advance(2)
    rival = await store_factory(h.store.path)
    expired, admission = await asyncio.gather(h.store.resolve_submission_expiry(event.task_id),
        rival.admit_with_policy(event,'v1',agent_id='assistant',max_inflight=64))
    assert expired == TaskResult(task_id=event.task_id,output=None,responding_agent=None,status='expired')
    assert admission.reason == 'expired'
    assert await h.store.get_task(event.task_id) is None
    assert await rival.resolve_submission_expiry(event.task_id) == expired
    assert await h.client.result(event.task_id) == expired
    assert await h.store.pending_runtime_count('assistant') == 0


async def test_unknown_client_task_returns_clear_error(runtime_harness):
    async with asyncio.timeout(.2):
        with pytest.raises(ValueError,match='unknown task'):
            await runtime_harness.client.result(str(uuid4()))


async def test_expiry_resolution_never_overrides_known_running_task(store,envelope_factory,fake_clock):
    event = envelope_factory(deadline=fake_clock.now()+timedelta(seconds=1))
    await store.submit(event)
    await store.admit_with_policy(event,'v1',agent_id='assistant',max_inflight=64)
    lease = await store.claim(event.task_id,OWNER,30)
    assert await store.start_handler(lease,retry_safe=False)
    fake_clock.advance(2)
    assert await store.resolve_submission_expiry(event.task_id) is None
    assert (await store.get_task(event.task_id)).status == 'running'
    fake_clock.advance(31)
    assert (await store.expire_task(event.task_id)).status == 'outcome_unknown'
    result = await store.resolve_submission_expiry(event.task_id)
    assert result.status == 'outcome_unknown' and result.responding_agent == 'assistant'


async def test_real_snapshot_admission_watermark_key_order_and_restart(runtime_harness,envelope_factory):
    from agentweave import Agent,SubscriptionPolicy
    from uuid import uuid5
    h = runtime_harness
    started, release = asyncio.Event(), asyncio.Event()
    applied = []
    agent = Agent('snapshots',concurrency=2,memory='stateful')
    @agent.on('snapshot',policy=SubscriptionPolicy(concurrency=2,queue_limit=4,overflow='latest',replaceable=True))
    async def apply(ctx,event):
        if ctx.event.source_version == 3:
            started.set()
            await release.wait()
        ctx.state['version'] = ctx.event.source_version
        applied.append(ctx.event.source_version)
        return ctx.event.source_version
    worker = await h.start(agent)
    three = envelope_factory(kind='snapshot',topic='snapshot',recipient=None,task_id=None,
        session_id=OWNER,key='k',source_version=3,payload={})
    await h.bus.publish(three)
    await asyncio.wait_for(started.wait(),20)
    two = three.model_copy(update={'event_id':str(uuid4()),'source_version':2})
    four = three.model_copy(update={'event_id':str(uuid4()),'source_version':4})
    await h.bus.publish(two)
    await h.bus.publish(four)
    three_id = str(uuid5(UUID(three.event_id),'task:snapshots'))
    four_id = str(uuid5(UUID(four.event_id),'task:snapshots'))
    async def admitted_four():
        return await h.store.get_task(four_id)
    try:
        assert (await eventually(admitted_four)).status in ('accepted','running')
        assert (await h.store.get_task(three_id)).status == 'running'
        assert applied == []
    finally:
        release.set()
    assert (await asyncio.wait_for(h.client.result(four_id),20)).output == 4
    assert applied == [3,4]
    assert (await h.store.get_task(three_id)).status == 'succeeded'
    assert (await h.store.load_session('snapshots',OWNER))[0] == {'version':4}
    replacement = await h.restart('snapshots')
    old = three.model_copy(update={'event_id':str(uuid4())})
    await h.bus.publish(old)
    async def stale():
        return replacement.metrics.counters['stale_version']
    await eventually(stale)
    assert applied == [3,4]


@pytest.mark.parametrize('asynchronous', [False, True])
async def test_executor_timeout_while_waiting_for_a_slot_means_not_started(asynchronous):
    # The limit still covers the wait for a slot, but expiring there is no
    # unknown effect: user code never ran, so it is HandlerNotStarted (expired),
    # not HandlerTimeout (outcome_unknown) as this test used to accept.
    import threading
    from agentweave._runtime.handlers import HandlerExecutor, HandlerNotStarted, HandlerTimeout
    executor = HandlerExecutor(1)
    started, release = threading.Event(),threading.Event()
    called = []
    def blocking(ctx,arg):
        started.set()
        release.wait(5)
        return arg
    def queued(ctx,arg):
        called.append(arg)
    async def queued_async(ctx,arg):
        called.append(arg)
    first = asyncio.create_task(executor.invoke(blocking,None,1,1))
    try:
        await asyncio.to_thread(started.wait,1)
        async with asyncio.timeout(.2):
            with pytest.raises(HandlerNotStarted) as raised:
                await executor.invoke(queued_async if asynchronous else queued,None,2,.01)
        assert not isinstance(raised.value, HandlerTimeout) and called == []
    finally:
        release.set()
        await first
        executor.close()


@pytest.mark.parametrize('asynchronous', [False, True])
async def test_executor_timeout_after_start_is_still_unknown(asynchronous):
    from agentweave._runtime.handlers import HandlerExecutor, HandlerTimeout
    import time
    executor = HandlerExecutor(1)
    def slow(ctx,arg):
        time.sleep(.1)
    async def slow_async(ctx,arg):
        await asyncio.sleep(1)
    try:
        with pytest.raises(HandlerTimeout):
            await executor.invoke(slow_async if asynchronous else slow,None,None,.02)
    finally:
        executor.close()


async def test_executor_rechecks_freshness_after_waiting_for_sync_slot():
    import threading
    from agentweave._runtime.handlers import HandlerExecutor
    executor = HandlerExecutor(1)
    started,release = threading.Event(),threading.Event()
    called = []
    class Freshness:
        def _assert_fresh(self):
            raise TimeoutError('input expired')
    def block(ctx,arg):
        started.set()
        release.wait(5)
    def late(ctx,arg):
        called.append(arg)
    first = asyncio.create_task(executor.invoke(block,None,None,1))
    try:
        await asyncio.to_thread(started.wait,1)
        queued = asyncio.create_task(executor.invoke(late,Freshness(),'late',1))
        await asyncio.sleep(.01)
        release.set()
        with pytest.raises(TimeoutError,match='expired'):
            await queued
        assert called == []
    finally:
        release.set()
        await first
        executor.close()


async def test_progress_iterator_reconnect_keeps_cursor_and_recovers_lost_notification(runtime_harness,monkeypatch):
    from agentweave import Agent
    h = runtime_harness
    entered,release = asyncio.Event(),asyncio.Event()
    agent = Agent('progressing')
    @agent.on_request
    async def long(ctx,request):
        entered.set()
        await release.wait()
        return 'done'
    await h.start(agent)
    submission = await h.client.agent('progressing').submit('x')
    await asyncio.wait_for(entered.wait(),20)
    original = await h.store.get_input_envelope(submission.task_id)
    subscriptions = []
    native = h.bus.subscribe
    async def subscribe(topics,group):
        subscription = await native(topics,group)
        if topics == [h.bus.control_topic('progress')]:
            subscriptions.append(subscription)
        return subscription
    monkeypatch.setattr(h.bus,'subscribe',subscribe)
    iterator = h.client.events(submission.task_id)
    first = original.model_copy(update={'event_id':str(uuid4()),'kind':'progress','topic':'progress','recipient':None,'payload':1})
    second = first.model_copy(update={'event_id':str(uuid4()),'payload':2})
    try:
        reading = asyncio.create_task(anext(iterator))
        async def subscribed():
            return subscriptions
        await eventually(subscribed)
        # Appending after subscription while the stream is starting exercises the read race.
        await h.store.append_progress(submission.task_id,first)
        await h.bus.publish(first)
        assert await asyncio.wait_for(reading,20) == first
        await subscriptions[-1].aclose()
        await h.store.append_progress(submission.task_id,second)  # no live notification
        assert await asyncio.wait_for(anext(iterator),20) == second
        assert len(subscriptions) >= 2
    finally:
        await iterator.aclose()
        release.set()


async def test_replacement_worker_reuses_busy_sync_pool(runtime_harness):
    import threading
    from agentweave import Agent
    from agentweave.definitions import RetryPolicy
    from agentweave._runtime.handlers import HandlerExecutor
    h = runtime_harness
    release = threading.Event()
    called = []
    agent = Agent('replace-sync')
    @agent.on_request(retry=RetryPolicy(timeout=.02))
    def blocked(ctx,request):
        called.append(request.input)
        release.wait(5)
        return request.input
    await h.start(agent)
    try:
        first = await h.ask('replace-sync','first')
        assert first.status == 'outcome_unknown'
        await h.restart('replace-sync')
        second = await h.ask('replace-sync','second')
        # The replacement shares the busy pool, so the second handler times out
        # waiting for the slot and never starts: expired, not outcome_unknown
        # (which this test used to accept although no code had run).
        assert second.status == 'expired'
        assert called == ['first']
    finally:
        release.set()


def test_native_executor_cleanup_survives_closed_event_loop():
    import threading
    from agentweave._runtime.handlers import HandlerExecutor
    from agentweave.errors import DefinitionError
    release = threading.Event()
    called = []
    owner = ('test-network','loop-close')
    def blocked(ctx,arg):
        called.append(arg)
        release.wait(5)
        return arg
    async def first_loop():
        executor = HandlerExecutor(1,owner=owner)
        try:
            with pytest.raises(TimeoutError):
                await executor.invoke(blocked,None,1,.01)
        finally:
            executor.close()
    try:
        asyncio.run(first_loop())
        with pytest.raises(DefinitionError,match='concurrency'):
            HandlerExecutor(2,owner=owner)
        async def second_loop():
            executor = HandlerExecutor(1,owner=owner)
            try:
                with pytest.raises(TimeoutError):
                    await executor.invoke(blocked,None,2,.01)
            finally:
                executor.close()
        asyncio.run(second_loop())
        assert called == [1]
    finally:
        release.set()
    async def cleanup():
        async with asyncio.timeout(2):
            while owner in HandlerExecutor._shared:
                await asyncio.sleep(.01)
        executor = HandlerExecutor(2,owner=owner)
        executor.close()
    asyncio.run(cleanup())


async def test_latency_starts_at_durable_admission_not_envelope_creation(runtime_harness,envelope_factory,fake_clock):
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = await h.start(echo_agent())
    event = envelope_factory(recipient='echo',payload='old-created',created_at=fake_clock.now()-timedelta(seconds=60))
    await h.store.submit(event)
    assert (await asyncio.wait_for(h.client.result(event.task_id),20)).status == 'succeeded'
    assert worker.metrics.result_latency == [0]
    assert (await h.store.get_task(event.task_id)).admitted_at == fake_clock.now()


async def test_client_connect_uses_existing_network_and_closes_only_owned_resources(broker_process,tmp_path):
    from agentweave import Client
    from agentweave.config import ProfileConfig,write_config
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.adapters.broker import BrokerBus
    from agentweave._directory import LiveDirectory
    from tests.fixtures.agents import echo_agent
    store = SQLiteStore(tmp_path/'prepared.sqlite3')
    config = ProfileConfig(root=tmp_path,profile='dev',project_id=str(uuid4()),network_id=str(uuid4()),
        broker_url=broker_process.url,broker_epoch=str(UUID(broker_process.epoch)),database_url='sqlite:///'+str(store.path))
    bus = BrokerBus(config.broker_url,config.network_id,profile='dev',broker_epoch=config.broker_epoch)
    directory = LiveDirectory(bus,store,config.network_id,broker_epoch=config.broker_epoch)
    harness = RuntimeHarness(config,store,bus,directory)
    try:
        await store.migrate()
        await harness.start(echo_agent())
        write_config(config,tmp_path/'.agentweave/dev/config.toml')
        async with Client.connect(tmp_path) as client:
            async def visible():
                return client.directory.snapshot().get('echo')
            await eventually(visible)
            assert (await asyncio.wait_for(client.agent('echo').ask('connected'),20)).output['input'] == 'connected'
        assert (await harness.ask('echo','still-running')).output['input'] == 'still-running'
    finally:
        await harness.close()
        await directory.close()
        await bus.close()
        await store.close()


async def test_event_only_worker_accepts_addressed_command_and_rejects_requests(runtime_harness,envelope_factory):
    from agentweave import Agent,SubscriptionPolicy
    from agentweave.errors import DefinitionError
    h = runtime_harness
    agent = Agent('command-only')
    @agent.on('command',policy=SubscriptionPolicy(overflow='latest',replaceable=True))
    async def command(ctx,event):
        return {'received':event,'key':ctx.event.key}
    await h.start(agent)
    with pytest.raises(DefinitionError,match='cannot accept requests'):
        await h.client.agent('command-only').submit('not-a-command')
    event = envelope_factory(recipient='command-only',kind='command',payload={'increment':1},key='k',source_version=1)
    await h.store.submit(event)
    result = await asyncio.wait_for(h.client.result(event.task_id),5)
    assert result.status == 'succeeded' and result.output == {'received':{'increment':1},'key':'k'}
    assert await h.store.get_input_envelope(event.task_id) == event


async def test_explicit_invalid_session_id_is_not_defaulted(runtime_harness):
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    await h.start(echo_agent())
    for invalid in ('',False,0):
        with pytest.raises(ValueError):
            await h.client.agent('echo').submit('x',session_id=invalid)


async def test_oversized_reply_fails_only_its_task_and_keeps_worker_serving(runtime_harness):
    from agentweave import Agent
    h = runtime_harness
    agent = Agent('oversized',memory='stateful')
    invocations = []
    @agent.on_request
    async def answer(ctx,request):
        invocations.append(request.input)
        if request.input == 'large':
            ctx.state['partial'] = True
            await ctx.publish('discarded.publication',{'partial':True})
            return 'x' * (1024 * 1024)
        return 'small'
    worker = await h.start(agent)
    submission = await h.client.agent('oversized').submit('large',session_id=OWNER)
    result = await asyncio.wait_for(h.client.result(submission.task_id),3)
    assert result.status == 'failed' and result.output is None
    assert await h.store.load_session('oversized',OWNER) == ({},[])
    import sqlite3
    with sqlite3.connect(h.store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox WHERE json_extract(envelope,'$.topic')='discarded.publication'").fetchone() == (0,)
    assert not h.runs['oversized'].done()
    assert (await h.ask('oversized','small',session_id=OWNER)).output == 'small'
    assert invocations == ['large','small']


async def test_async_timeout_ignores_cancellation_suppression_and_retains_ownership():
    from agentweave._runtime.handlers import HandlerExecutor
    class Context:
        _active = True
        _lease_valid = True
        def _assert_fresh(self):
            pass
    release,cancelled = asyncio.Event(),asyncio.Event()
    calls = []
    owner = ('fix1','delaying-async')
    first = HandlerExecutor(1,owner=owner)
    context = Context()
    async def opaque(ctx,arg):
        calls.append(arg)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return 'late'
    invocation = asyncio.create_task(first.invoke(opaque,context,'first',.01))
    second = None
    try:
        completed,_ = await asyncio.wait([invocation],timeout=.15)
        assert completed, 'configured timeout waited for opaque cancellation cooperation'
        with pytest.raises(TimeoutError):
            await invocation
        await asyncio.wait_for(cancelled.wait(),.2)
        assert not context._active and not context._lease_valid
        first.close()
        second = HandlerExecutor(1,owner=owner)
        with pytest.raises(TimeoutError):
            await second.invoke(opaque,Context(),'replacement',.01)
        assert calls == ['first']
        assert second.state.outstanding == 1
    finally:
        release.set()
        await asyncio.gather(invocation,return_exceptions=True)
        first.close()
        if second is not None:
            second.close()
        async def cleaned():
            return owner not in HandlerExecutor._shared
        await eventually(cleaned,timeout=2)


async def test_async_timeout_seals_context_and_ignores_late_state(runtime_harness):
    from agentweave import Agent
    from agentweave.definitions import RetryPolicy
    h = runtime_harness
    release,cancelled,finished = asyncio.Event(),asyncio.Event(),asyncio.Event()
    contexts,late_errors = [],[]
    agent = Agent('late-async',memory='stateful')
    @agent.on_request(retry=RetryPolicy(timeout=.02))
    async def delayed(ctx,request):
        contexts.append(ctx)
        ctx.state['partial'] = True
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        ctx.state['late'] = True
        try:
            await ctx.publish('too.late',{'late':True})
        except Exception as error:
            late_errors.append(error)
        finished.set()
        return 'late'
    worker = await h.start(agent)
    submission = await h.client.agent(agent.name).submit('x',session_id=OWNER)
    try:
        result = await asyncio.wait_for(h.client.result(submission.task_id),2)
        assert result.status == 'outcome_unknown'
        assert not contexts[0]._active and not contexts[0]._lease_valid
        await asyncio.wait_for(cancelled.wait(),1)
        assert worker.engine.executor.state.outstanding == 1
        release.set()
        await asyncio.wait_for(finished.wait(),1)
        assert late_errors and not contexts[0]._outgoing
        assert await h.store.load_session(agent.name,OWNER) == ({},[])
        assert (await h.store.get_task(submission.task_id)).status == 'outcome_unknown'
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(),2)


async def test_stop_finishes_without_waiting_for_opaque_async_cancellation(runtime_harness):
    from agentweave import Agent
    h = runtime_harness
    entered,release,cancelled,finished = [asyncio.Event() for _ in range(4)]
    contexts = []
    agent = Agent('stop-delaying')
    @agent.on_request
    async def delayed(ctx,request):
        contexts.append(ctx)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        finished.set()
        return 'late'
    worker = await h.start(agent)
    submission = await h.client.agent(agent.name).submit('x')
    await asyncio.wait_for(entered.wait(),5)
    stopping = asyncio.create_task(worker.stop())
    try:
        completed,_ = await asyncio.wait([stopping],timeout=1)
        assert completed, 'stop waited for opaque cancellation cooperation'
        await stopping
        assert worker._done.is_set()
        assert not contexts[0]._active and not contexts[0]._lease_valid
        await asyncio.wait_for(cancelled.wait(),1)
        assert worker.engine.executor.state.outstanding == 1
        assert (await h.store.get_task(submission.task_id)).status == 'running'
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(),2)
        await asyncio.gather(stopping,return_exceptions=True)


async def test_cleanup_failure_always_signals_done_and_is_visible(runtime_harness,monkeypatch):
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = await h.start(echo_agent('cleanup-error'))
    native = h.directory.withdraw
    async def broken(*args):
        raise RuntimeError('deliberate cleanup failure')
    monkeypatch.setattr(h.directory,'withdraw',broken)
    stopping = asyncio.create_task(worker.stop())
    try:
        completed,_ = await asyncio.wait([stopping],timeout=.5)
        assert completed, 'cleanup exception stranded stop waiter'
        with pytest.raises(RuntimeError,match='deliberate cleanup failure'):
            await stopping
        assert worker._done.is_set()
        with pytest.raises(RuntimeError,match='deliberate cleanup failure'):
            await h.runs[worker.definition.name]
    finally:
        monkeypatch.setattr(h.directory,'withdraw',native)
        if not stopping.done():
            worker._done.set()  # regression cleanup only: release the old broken implementation's waiter
        await asyncio.gather(stopping,h.runs[worker.definition.name],return_exceptions=True)
        await native(worker.definition.name,worker.instance_id)
        h.workers.pop(worker.definition.name)
        h.runs.pop(worker.definition.name)


async def test_shutdown_cleanup_uses_remaining_budget_even_if_aclose_ignores_cancel(runtime_harness,monkeypatch):
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = await h.start(echo_agent('cleanup-bound'))
    subscription = worker._subs[0]
    native = subscription.aclose
    entered,release = asyncio.Event(),asyncio.Event()
    async def delayed():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        await native()
    monkeypatch.setattr(subscription,'aclose',delayed)
    worker._shutdown_deadline = asyncio.get_running_loop().time()+.05
    stopping = asyncio.create_task(worker.stop())
    try:
        await asyncio.wait_for(entered.wait(),.5)
        completed,_ = await asyncio.wait([stopping],timeout=.3)
        assert completed, 'cleanup exceeded its remaining monotonic shutdown budget'
        with pytest.raises(TimeoutError):
            await stopping
        assert worker._done.is_set()
    finally:
        release.set()
        await asyncio.gather(stopping,h.runs[worker.definition.name],return_exceptions=True)
        pending = list(getattr(worker,'_cleanup_jobs',()))
        if pending:
            await asyncio.gather(*pending,return_exceptions=True)
        monkeypatch.setattr(subscription,'aclose',native)
        h.workers.pop(worker.definition.name)
        h.runs.pop(worker.definition.name)


async def test_drain_and_cleanup_share_one_remaining_monotonic_budget(runtime_harness):
    from agentweave import Agent
    h = runtime_harness
    entered,release,cancelled,finished = [asyncio.Event() for _ in range(4)]
    contexts = []
    agent = Agent('drain-delaying')
    @agent.on_request
    async def delayed(ctx,request):
        contexts.append(ctx)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        finished.set()
        return 'late'
    worker = await h.start(agent)
    submission = await h.client.agent(agent.name).submit('x')
    await asyncio.wait_for(entered.wait(),5)
    loop = asyncio.get_running_loop()
    worker._shutdown_deadline = loop.time()+.05  # remaining budget after an earlier part of drain
    before = loop.time()
    draining = asyncio.create_task(worker.drain())
    try:
        completed,_ = await asyncio.wait([draining],timeout=.3)
        assert completed, 'drain gave cleanup a new grace period'
        with pytest.raises(TimeoutError):
            await draining
        assert loop.time()-before < .3
        assert worker._done.is_set()
        assert not contexts[0]._active and not contexts[0]._lease_valid
        await asyncio.wait_for(cancelled.wait(),.5)
        assert worker.engine.executor.state.outstanding == 1
        assert (await h.store.get_task(submission.task_id)).status == 'running'
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(),2)
        await asyncio.gather(draining,h.runs[agent.name],return_exceptions=True)
        if worker._cleanup_jobs:
            await asyncio.gather(*list(worker._cleanup_jobs),return_exceptions=True)
        h.workers.pop(agent.name)
        h.runs.pop(agent.name)


async def test_sync_returned_awaitable_uses_the_same_independent_timeout():
    from agentweave._runtime.handlers import HandlerExecutor
    release,cancelled = asyncio.Event(),asyncio.Event()
    executor = HandlerExecutor(1)
    async def late():
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return 'late'
    def handler(ctx,arg):
        return late()
    invocation = asyncio.create_task(executor.invoke(handler,None,None,.02))
    try:
        completed,_ = await asyncio.wait([invocation],timeout=.2)
        assert completed
        with pytest.raises(TimeoutError):
            await invocation
        await asyncio.wait_for(cancelled.wait(),.5)
        assert executor.state.outstanding == 1
    finally:
        release.set()
        await asyncio.gather(invocation,return_exceptions=True)
        async def completed():
            return executor.state.outstanding == 0
        await eventually(completed,timeout=2)
        executor.close()


async def test_stop_before_run_releases_ownership_without_directory_io(runtime_harness):
    from agentweave import Worker
    from agentweave._runtime.handlers import HandlerExecutor
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = Worker(echo_agent('never-started'),h.config,h.store,h.bus,h.directory)
    await worker.stop()
    await worker.stop()
    assert worker._done.is_set()
    assert not h.directory._started
    assert (h.config.network_id,'never-started') not in HandlerExecutor._shared


@pytest.mark.parametrize('phase',['start','publish','receipt','inbox'])
@pytest.mark.parametrize('exhaust_budget',[False,True])
async def test_stop_tracks_late_startup_ownership(runtime_harness,monkeypatch,phase,exhaust_budget):
    from agentweave import Worker
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = Worker(echo_agent('late-startup'),h.config,h.store,h.bus,h.directory)
    entered,release = asyncio.Event(),asyncio.Event()
    acquired = []
    native_start,native_publish,native_subscribe = h.directory.start,h.directory.publish,h.bus.subscribe
    calls = 0
    async def delay():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
    async def start():
        if phase == 'start':
            await delay()
        await native_start()
    async def publish(participant):
        if phase == 'publish':
            await delay()
        await native_publish(participant)
    async def subscribe(*args,**kwargs):
        nonlocal calls
        calls += 1
        # Directory.start also subscribes; only delay the Worker's own calls.
        if ((phase == 'receipt' and args[0] == [h.bus.control_topic('receipt')])
                or (phase == 'inbox' and args[0] == [h.bus.agent_topic(worker.definition.name)])):
            await delay()
        sub = await native_subscribe(*args,**kwargs)
        if args[0] != [h.bus.control_topic('directory')]:
            acquired.append(sub)
        return sub
    monkeypatch.setattr(h.directory,'start',start)
    monkeypatch.setattr(h.directory,'publish',publish)
    monkeypatch.setattr(h.bus,'subscribe',subscribe)
    running = asyncio.create_task(worker.run())
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(),3)
        worker._shutdown_deadline = asyncio.get_running_loop().time()+(.05 if exhaust_budget else 2)
        stopping = asyncio.create_task(worker.stop())
        if exhaust_budget:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(stopping),.5)
            assert worker._done.is_set()
            assert worker._cleanup_jobs, 'unresolved startup ownership was forgotten'
        else:
            completed,_ = await asyncio.wait([stopping],timeout=.02)
            assert not completed, 'stop reported success before startup ownership was resolved'
        release.set()
        await asyncio.gather(running,return_exceptions=True)
        if not exhaust_budget:
            await stopping
        async def released():
            return (not worker._cleanup_jobs and all(sub.closed for sub in acquired)
                    and (worker.definition.name,worker.instance_id) not in h.directory._owned)
        await eventually(released,timeout=3)
        assert not worker.ready.is_set()
        assert all(job.done() for job in worker._jobs)
        assert calls <= (1 if phase == 'start' else 2 if phase in ('publish','receipt') else 3)
    finally:
        release.set()
        await asyncio.gather(running,*([stopping] if stopping else []),return_exceptions=True)
        for job in worker._jobs:
            job.cancel()
        await asyncio.gather(*worker._jobs,return_exceptions=True)
        pending = list(worker._cleanup_jobs)
        if pending:
            await asyncio.gather(*pending,return_exceptions=True)
        for sub in acquired:
            await sub.aclose()
        if h.directory._started:
            await h.directory.withdraw(worker.definition.name,worker.instance_id)


@pytest.mark.parametrize('enter_runner',[False,True])
async def test_stop_racing_first_startup_scheduling(runtime_harness,enter_runner):
    from agentweave import Worker
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = Worker(echo_agent('startup-scheduling'),h.config,h.store,h.bus,h.directory)
    running = asyncio.create_task(worker.run())
    if enter_runner:
        await asyncio.sleep(0)  # run entered; its acquisition task has not yet run
    await worker.stop()
    outcome = (await asyncio.gather(running,return_exceptions=True))[0]
    assert isinstance(outcome,(RuntimeError,asyncio.CancelledError))
    assert worker._done.is_set() and not worker.ready.is_set()
    assert not worker._cleanup_jobs and not worker._jobs
    assert not h.directory._started


async def test_failed_late_startup_compensation_retains_owned_resource(runtime_harness,monkeypatch):
    from agentweave import Worker
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = Worker(echo_agent('startup-compensation-error'),h.config,h.store,h.bus,h.directory)
    entered,release,failed = asyncio.Event(),asyncio.Event(),asyncio.Event()
    native_subscribe = h.bus.subscribe
    retained = []
    async def subscribe(*args,**kwargs):
        if args[0] != [h.bus.control_topic('receipt')]:
            return await native_subscribe(*args,**kwargs)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        sub = await native_subscribe(*args,**kwargs)
        native_close = sub.aclose
        async def broken():
            failed.set()
            raise RuntimeError('deliberate late compensation failure')
        monkeypatch.setattr(sub,'aclose',broken)
        retained.append((sub,native_close))
        return sub
    monkeypatch.setattr(h.bus,'subscribe',subscribe)
    running = asyncio.create_task(worker.run())
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(),3)
        worker._shutdown_deadline = asyncio.get_running_loop().time()+.03
        stopping = asyncio.create_task(worker.stop())
        with pytest.raises(TimeoutError):
            await stopping
        release.set()
        await asyncio.wait_for(failed.wait(),3)
        await asyncio.gather(running,return_exceptions=True)
        await asyncio.sleep(0)
        assert any('deliberate late compensation failure' in str(e) for e in worker._cleanup_errors)
        assert retained[0][0] in worker._startup_unreleased.values()
        assert not worker.ready.is_set() and all(job.done() for job in worker._jobs)
        with pytest.raises(TimeoutError):
            await worker.stop()  # no new grace or automatic compensation retry
    finally:
        release.set()
        await asyncio.gather(running,*([stopping] if stopping else []),return_exceptions=True)
        if worker._cleanup_jobs:
            await asyncio.gather(*list(worker._cleanup_jobs),return_exceptions=True)
        for sub,native_close in retained:
            monkeypatch.setattr(sub,'aclose',native_close)
            await native_close()


@pytest.mark.parametrize('outcome',['within_grace','deadline_cooperative','deadline_opaque'])
async def test_partial_publication_compensation_keeps_its_deadline_owner(runtime_harness,monkeypatch,outcome):
    from agentweave import Worker
    from tests.fixtures.agents import echo_agent
    h = runtime_harness
    worker = Worker(echo_agent('partial-publication'),h.config,h.store,h.bus,h.directory)
    entered,release,cancelled,finished = [asyncio.Event() for _ in range(4)]
    native_publish,native_withdraw = h.directory.publish,h.directory.withdraw
    async def publish(participant):
        await native_publish(participant)
        worker._shutdown_deadline = asyncio.get_running_loop().time()+(2 if outcome == 'within_grace' else .1)
        raise RuntimeError('controlled partial publication failure')
    async def withdraw(*args):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            if outcome != 'deadline_opaque':
                raise
            await release.wait()
        await native_withdraw(*args)
        finished.set()
    monkeypatch.setattr(h.directory,'publish',publish)
    monkeypatch.setattr(h.directory,'withdraw',withdraw)
    running = asyncio.create_task(worker.run())
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(),3)
        assert (worker.definition.name,worker.instance_id) in h.directory._owned
        stopping = asyncio.create_task(worker.stop())
        if outcome == 'within_grace':
            completed,_ = await asyncio.wait([stopping],timeout=.02)
            assert not completed, 'shutdown resolved while publication compensation was still outstanding'
            release.set()
            with pytest.raises(RuntimeError,match='controlled partial publication failure'):
                await asyncio.wait_for(asyncio.shield(stopping),1)
            assert finished.is_set()
            assert (worker.definition.name,worker.instance_id) not in h.directory._owned
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(stopping),.5)
            await asyncio.wait_for(cancelled.wait(),.3)
            assert worker._done.is_set()
            assert (worker.definition.name,worker.instance_id) in h.directory._owned
            assert worker._startup_unreleased
            if outcome == 'deadline_opaque':
                assert any(not job.done() for job in worker._cleanup_jobs)
                release.set()
                await asyncio.wait_for(finished.wait(),1)
                async def released():
                    return not worker._cleanup_jobs and not worker._startup_unreleased
                await eventually(released,timeout=1)
            with pytest.raises(TimeoutError):
                await worker.stop()  # same public deadline outcome, no new compensation window
        run_error = (await asyncio.gather(running,return_exceptions=True))[0]
        def includes_publication_error(error):
            return ('controlled partial publication failure' in str(error)
                    or isinstance(error,BaseExceptionGroup) and any(includes_publication_error(e) for e in error.exceptions))
        assert includes_publication_error(run_error)
        assert not worker.ready.is_set() and not worker._jobs
    finally:
        release.set()
        await asyncio.gather(running,*([stopping] if stopping else []),return_exceptions=True)
        if worker._cleanup_jobs:
            await asyncio.gather(*list(worker._cleanup_jobs),return_exceptions=True)
        monkeypatch.setattr(h.directory,'withdraw',native_withdraw)
        if h.directory._started:
            await native_withdraw(worker.definition.name,worker.instance_id)
