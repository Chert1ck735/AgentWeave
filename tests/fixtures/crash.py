"""Worker processes with failpoints: a real store, bus and Worker, killed at one exact point.

A failpoint wraps one journal or bus operation in the child process and ends
the process with os._exit right after that operation committed, as a power
loss would. It fires once per test (a marker file), so a restarted worker
runs normally. External effects are lines appended to a file, so a test
counts how often an effect really happened across processes.
"""
import json
import os
import time
from pathlib import Path

CRASH_EXIT = 137


class Failpoint:
    def __init__(self, name, marker):
        self.name, self.marker = name, Path(marker)

    def fire(self, name):
        """End the process here, once, if this is the armed failpoint."""
        if name != self.name or self.marker.exists():
            return
        self.marker.write_text(name)
        os._exit(CRASH_EXIT)

    def after(self, owner, method, name, when=lambda *args, **kwargs: True):
        original = getattr(owner, method)

        async def wrapped(*args, **kwargs):
            result = await original(*args, **kwargs)
            if when(result, *args, **kwargs):
                self.fire(name)
            return result
        setattr(owner, method, wrapped)


def record(effects, kind, **details):
    """One external effect, durable before the call returns."""
    with open(effects, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'kind': kind, 'pid': os.getpid(), 'at': time.time()} | details) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def effects_of(effects, kind=None):
    path = Path(effects)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if kind is None or row['kind'] == kind]


class ScriptModel:
    """Calls the record tool once, then answers: deterministic across restarts and replays."""

    def __init__(self, options, failpoint):
        self.options, self.failpoint = options, failpoint

    def capabilities(self):
        from agentweave.models import ModelCapabilities
        return ModelCapabilities()

    def validate_request(self, request):
        from agentweave.providers import check_capabilities
        check_capabilities(self.capabilities(), request)

    async def generate(self, request):
        import asyncio
        from agentweave.models import TurnComplete
        from agentweave.testing import text_turn, tool_turn
        if self.options.get('model_effects'):
            record(self.options['model_effects'], 'model_start')
        self.failpoint.fire('in_model')
        if self.options.get('model_seconds'):
            await asyncio.sleep(self.options['model_seconds'])
        if self.options.get('model_effects'):
            record(self.options['model_effects'], 'model_end')
        answered = any(message.role == 'tool' for message in request.messages)
        turn = text_turn('done') if answered or self.options.get('no_tools') else tool_turn('record_effect', {})
        yield TurnComplete(turn=turn)

    async def close(self):
        return None


def build_agent(kind, options, failpoint):
    from agentweave import Agent, Model, Peers, tool
    effects = options['effects']
    if kind == 'loop':
        @tool(retry='safe' if options.get('safe_tool') else None)
        def record_effect() -> str:
            """Perform the external effect."""
            record(effects, 'tool')
            failpoint.fire('tool_effect')
            return 'recorded'
        return Agent(options.get('name', 'loop'), model=Model.custom(ScriptModel(options, failpoint)),
                     tools=[] if options.get('no_tools') else [record_effect])
    if kind == 'general':
        agent = Agent('general', peers=Peers.allow(['analyst']))

        @agent.on_request
        async def route(ctx, request):
            ctx.handoff('analyst', message='please answer', scope='task')
        return agent
    if kind == 'asker':
        agent = Agent('asker', peers=Peers.allow(['analyst']))

        @agent.on_request(retry='safe')
        async def ask(ctx, request):
            answer = await ctx.consult('analyst', request.input)
            return {'consulted': answer.output}
        return agent
    if kind == 'analyst':
        agent = Agent('analyst')

        @agent.on_request
        async def answer(ctx, request):
            record(effects, 'analyst')
            return 'analysed'
        return agent
    if kind == 'slow':
        agent = Agent('slow', concurrency=options.get('concurrency', 1))

        @agent.on_request(retry='safe')
        async def work(ctx, request):
            import asyncio
            record(effects, 'start', task=ctx.task_id)
            await asyncio.sleep(options.get('seconds', 0.3))
            record(effects, 'end', task=ctx.task_id)
            return os.getpid()
        return agent
    if kind == 'blocking':
        agent = Agent('blocking')

        @agent.on_request(retry='safe')
        async def wait(ctx, request):
            import asyncio
            record(effects, 'start', task=ctx.task_id)
            while not Path(options['release']).exists():
                await asyncio.sleep(0.05)
            return os.getpid()
        return agent
    if kind == 'listener':
        agent = Agent('listener')

        @agent.on('work.item')
        async def handle(ctx, event):
            record(effects, 'listener', event=ctx.event.event_id)
            return 'handled'
        return agent
    raise ValueError(kind)


def install_outage(store, flag):
    """While flag exists, every journal operation fails as an unreachable database would."""
    from agentweave.errors import StoreUnavailable
    original = store._retry

    async def unavailable(operation):
        if Path(flag).exists():
            raise StoreUnavailable('the journal is unreachable (test outage)')
        return await original(operation)
    store._retry = unavailable


def install(failpoint, store, bus):
    """Arm every journal and bus failpoint; only the named one ever fires."""
    from agentweave._records import EventRejection
    failpoint.after(store, 'admit_with_policy', 'input_commit',
                    lambda result, *a, **k: not isinstance(result, EventRejection) and not result.duplicate)
    failpoint.after(store, 'finish_model_attempt', 'model_response')
    failpoint.after(store, 'commit_step', 'step_commit', lambda result, *a, **k: result.status == 'succeeded')
    from agentweave._runtime.delivery import AdmissionService

    def consult_receipt(result, service, delivery, *a, **k):
        # Only the receipt a consulted peer (analyst) sends for this agent's consult.
        event = delivery.envelope
        return event.kind == 'receipt' and isinstance(event.payload, dict) and event.payload.get('recipient') == 'analyst'
    failpoint.after(AdmissionService, 'accept_receipt', 'receipt_commit', consult_receipt)
    failpoint.after(store, 'transfer', 'ownership_transfer')
    # The client's request, published from the shared outbox by this worker's pump.
    failpoint.after(bus, 'publish', 'outbox_publish', lambda result, envelope, *a, **k: envelope.kind == 'request')


def run_crash_worker(config_json, path, clock_path, kind, options_json, ready, stop):
    """A spawned worker with one armed failpoint; reports ('ready', kind, instance_id) once serving."""
    import asyncio
    from agentweave import Worker
    from agentweave.config import ProfileConfig
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.adapters.broker import BrokerBus
    from agentweave._directory import LiveDirectory
    from tests.fixtures.agents import FileClock
    options = json.loads(options_json)

    async def run():
        config = ProfileConfig.model_validate(json.loads(config_json))
        clock = FileClock(clock_path)
        store = SQLiteStore(path, clock=clock)
        bus = BrokerBus(config.broker_url, config.network_id, profile='dev', broker_epoch=config.broker_epoch)
        directory = LiveDirectory(bus, store, config.network_id, broker_epoch=config.broker_epoch, clock=clock)
        failpoint = Failpoint(options.get('failpoint'), options['marker'])
        install(failpoint, store, bus)
        if options.get('outage'):
            install_outage(store, options['outage'])
        worker = Worker(build_agent(kind, options, failpoint), config, store, bus, directory)
        running = None
        try:
            await store.migrate()
            running = asyncio.create_task(worker.run())
            waiting = asyncio.create_task(worker.ready.wait())
            done, _ = await asyncio.wait([running, waiting], timeout=20, return_when=asyncio.FIRST_COMPLETED)
            if running in done:
                await running
            if waiting not in done:
                raise TimeoutError('worker startup')
            ready.put(('ready', kind, worker.instance_id))
            while not stop.is_set():
                if running.done():
                    await running
                    break
                await asyncio.sleep(.02)
            await worker.stop()
            await running
        except Exception as exc:
            causes = [type(item).__name__ for item in getattr(exc, 'exceptions', ())]
            ready.put(('error', kind, f'{type(exc).__name__}: {exc} {causes}'))
        finally:
            if running is not None and not running.done():
                await worker.stop()
                await asyncio.gather(running, return_exceptions=True)
            # As RuntimeResources.close: a failing withdrawal still closes the bus threads and the store.
            try:
                await directory.close()
            finally:
                try:
                    await bus.close()
                finally:
                    await store.close()
    asyncio.run(run())
