"""An AI assistant with a durable calculator tool and a token budget.

Start (needs OPENAI_API_KEY and a model ID you choose):
    AGENTWEAVE_MODEL=<model-id> agentweave dev run examples.assistant:assistant --detach
Ask:
    AGENTWEAVE_MODEL=<model-id> python -m examples.assistant "What is 12 + 7?"
Without a key, python -m examples.assistant --offline runs the same loop
in-process with a scripted FakeModel and no network.
"""
import asyncio
import os
import sys
from agentweave import Agent, Client, Model, tool
from agentweave.models import ModelBudget

ANSWER_TIMEOUT_SECONDS = 120


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def build(model) -> Agent:
    return Agent('assistant', instructions='Answer briefly. Use the calculator for arithmetic.',
                 tools=[add], model=model, memory='stateful', budget=ModelBudget(max_tokens=20_000))


assistant = build(Model.openai(os.environ.get('AGENTWEAVE_MODEL', 'set-AGENTWEAVE_MODEL')))


async def ask(question: str) -> None:
    async with Client.connect() as client:
        async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
            result = await client.agent('assistant').ask(question)
        print(result.status, result.output)


async def offline(question: str) -> None:
    """Run one loop against a scripted model and a temporary journal."""
    import tempfile
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4
    from agentweave._runtime.steps import ModelLoop
    from agentweave._runtime.tools import ToolExecutor
    from agentweave._records import SystemClock
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.messages import Envelope
    from agentweave.testing import FakeModel, text_turn, tool_turn
    model = FakeModel([tool_turn('add', {'a': 12, 'b': 7}), text_turn('12 + 7 = 19')])
    agent = build(Model.custom(model))
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteStore(os.path.join(directory, 'journal.sqlite3'))
        await store.migrate()
        now, instance = datetime.now(timezone.utc), str(uuid4())
        event = Envelope(event_id=str(uuid4()), network_id=str(uuid4()), kind='request', sender='client.offline',
                         recipient='assistant', topic=None, task_id=str(uuid4()), parent_task_id=None,
                         correlation_id=None, session_id=str(uuid4()), created_at=now,
                         deadline=now + timedelta(seconds=ANSWER_TIMEOUT_SECONDS), traceparent=None, payload=question)
        await store.admit(event, 'offline')
        lease = await store.claim(event.task_id, instance, 30)
        task = await store.get_task(event.task_id)
        tools = ToolExecutor(agent, store, instance_id=instance, clock=SystemClock())
        loop = ModelLoop(agent, store, model, instance_id=instance, clock=SystemClock(), tools=tools)
        try:
            outcome = await loop.run(task, lease)
        finally:
            loop.close()
            await store.close()
    print(outcome.status, outcome.output)


if __name__ == '__main__':
    arguments = [value for value in sys.argv[1:] if value != '--offline']
    question = ' '.join(arguments) or 'What is 12 + 7?'
    asyncio.run(offline(question) if '--offline' in sys.argv else ask(question))
