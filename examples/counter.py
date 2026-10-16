"""A stateful Python counter on the local dev network.

Start: agentweave dev run examples.counter:counter --detach
Request and event demonstration: python -m examples.counter
Stop worker: agentweave agent stop counter
Stop broker (journal preserved): agentweave infra down
"""
import asyncio
from uuid import uuid4
from agentweave import Agent, Client

RESULT_TIMEOUT_SECONDS = 120

counter = Agent('counter', memory='stateful')

@counter.on_request(retry='safe')
async def request_count(ctx, request):
    ctx.state['count'] = ctx.state.get('count', 0) + len(request.input)
    return ctx.state['count']

@counter.on('numbers.received', retry='safe')
async def event_count(ctx, event):
    ctx.state['count'] = ctx.state.get('count', 0) + len(event['numbers'])
    return ctx.state['count']

async def main():
    session_id = str(uuid4())
    async with Client.connect() as client:
        print((await client.agent('counter').ask([1], session_id)).output)
        event_id = await client.publish('numbers.received', {'numbers': [2, 3]}, session_id=session_id)
        async with asyncio.timeout(RESULT_TIMEOUT_SECONDS):
            print((await client.event_result(event_id, 'counter')).output)

if __name__ == '__main__':
    asyncio.run(main())
