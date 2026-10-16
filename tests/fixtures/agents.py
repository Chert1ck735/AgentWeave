"""Stable plain Python definitions also imported by spawned worker processes."""
from agentweave import Agent, SubscriptionPolicy


def counter_agent():
    agent = Agent('counter', memory='stateful')
    @agent.on_request(retry='safe')
    async def request(ctx, request):
        ctx.state['count'] = ctx.state.get('count', 0) + len(request.input)
        await ctx.publish('count.updated', {'count':ctx.state['count']}, key='counter', source_version=ctx.state['count'])
        return ctx.state['count']
    @agent.on('numbers.received', retry='safe')
    async def event(ctx, event):
        ctx.state['count'] = ctx.state.get('count', 0) + len(event['numbers'])
        return ctx.state['count']
    return agent


def echo_agent(name='echo'):
    agent = Agent(name)
    @agent.on_request
    async def echo(ctx, request):
        ctx.state['temporary'] = True
        return {'input':request.input, 'history':len(request.history)}
    return agent


def crash_agent(retry_safe):
    from pathlib import Path
    import asyncio
    agent = Agent('crash',memory='stateful')
    @agent.on_request(retry='safe' if retry_safe else None)
    async def perform(ctx,request):
        previous = ctx.state.get('count',0)
        ctx.state['count'] = previous + 1
        if request.input.get('marker'):
            marker = Path(request.input['marker'])
            if not marker.exists():
                marker.write_text('started')
                ctx.state['count'] = 10000  # never part of a committed snapshot
                await asyncio.Event().wait()
        return ctx.state['count']
    return agent


class FileClock:
    def __init__(self,path):
        self.path = path
    def now(self):
        from datetime import datetime
        from pathlib import Path
        return datetime.fromisoformat(Path(self.path).read_text())


def run_worker_process(config_json,path,clock_path,kind,retry_safe,ready,stop):
    """Spawned processes use the real store, bus and Worker without mocks."""
    import asyncio
    from agentweave import Worker
    from agentweave.config import ProfileConfig
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.adapters.broker import BrokerBus
    from agentweave._directory import LiveDirectory
    async def run():
        import json
        config = ProfileConfig.model_validate(json.loads(config_json))
        clock = FileClock(clock_path)
        store = SQLiteStore(path,clock=clock)
        bus = BrokerBus(config.broker_url,config.network_id,profile='dev',broker_epoch=config.broker_epoch)
        directory = LiveDirectory(bus,store,config.network_id,broker_epoch=config.broker_epoch,clock=clock)
        definition = crash_agent(retry_safe) if kind == 'crash' else counter_agent()
        worker = Worker(definition,config,store,bus,directory)
        running = None
        try:
            await store.migrate()
            running = asyncio.create_task(worker.run())
            waiting = asyncio.create_task(worker.ready.wait())
            done,_ = await asyncio.wait([running,waiting],timeout=20,return_when=asyncio.FIRST_COMPLETED)
            if running in done:
                await running
            if waiting not in done:
                raise TimeoutError('worker startup')
            ready.put(('ready',worker.instance_id))
            while not stop.is_set():
                if running.done():
                    await running
                    break
                await asyncio.sleep(.02)
            await worker.stop()
            await running
        except Exception as exc:
            ready.put(('error',type(exc).__name__,str(exc)))
        finally:
            if running is not None and not running.done():
                await worker.stop()
                await asyncio.gather(running,return_exceptions=True)
            await directory.close()
            await bus.close()
            await store.close()
    asyncio.run(run())
