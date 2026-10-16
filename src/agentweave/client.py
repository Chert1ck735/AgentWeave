"""Trusted project requests over already configured infrastructure."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
from ._records import SystemClock
from .config import resolve
from .errors import BrokerUnavailable, ConfigError, ConversationNotFound, DefinitionError
from .messages import (CLIENT_PREFIX, Envelope, JSONValue, TaskResult, TaskSubmission, event_task_id,
                       validate_event_topic, validate_name)

_TERMINAL = frozenset(('succeeded','failed','cancelled','expired','outcome_unknown'))
RESULT_POLL_SECONDS = 0.02
# A long wait polls less often: many waiting clients must not saturate their own journal connection.
RESULT_POLL_MAX_SECONDS = 0.25
RESULT_POLL_GROWTH = 1.5


class Client:
    def __init__(self, config, store, bus, directory):
        self.config, self.store, self.bus, self.directory = config, store, bus, directory
        self.clock = getattr(store, 'clock', SystemClock())
        self.sender = CLIENT_PREFIX + uuid4().hex
        self._closed = False

    @classmethod
    @asynccontextmanager
    async def connect(cls, root=None, profile='dev'):
        from ._infrastructure.bootstrap import Bootstrap
        from ._infrastructure.processes import ProcessRegistry
        root = Path(root or Path.cwd()).resolve()
        try:
            config = resolve(root,profile)
        except ConfigError:
            config = None
        managed = bool(ProcessRegistry(root,profile).records()) or (root/'.agentweave'/profile/'managed.json').exists()
        prepared = (config is not None and config.broker_url and config.database_url
                    and (profile != 'dev' or config.broker_epoch))
        if managed or not prepared:
            resources = await Bootstrap(root,profile).ensure(config)
        else:
            # Explicit prepared connections grant connection authority only.
            resources = await Bootstrap.open_resources(config)
        client = cls(resources.config,resources.store,resources.bus,resources.directory)
        try:
            yield client
        finally:
            await client.close()
            await resources.close()


    async def conversation(self, agent: str, conversation_id: str | None = None) -> 'Conversation':
        """Open a conversation answered by agent, or reattach to an existing conversation_id.

        A handoff inside it moves the conversation: the next message goes to its
        new active agent. The owner is always read from the journal.
        """
        if self._closed:
            raise RuntimeError('Client is closed')
        if conversation_id is not None:
            # Reattaching: the journal names who answers now; agent is only the opener of a new one.
            record = await self.store.get_conversation(conversation_id)
            if record is None:
                raise ConversationNotFound(f'no conversation {conversation_id}')
        else:
            self._target(validate_name(agent))
            record = await self.store.open_conversation(agent)
        return Conversation(self, record.session_id, record.active_agent)

    def agent(self, name: str) -> 'AgentClient':
        if self._closed:
            raise RuntimeError('Client is closed')
        return AgentClient(self, validate_name(name))

    def _target(self, name):
        if self._closed:
            raise RuntimeError('Client is closed')
        self.directory.check_health()
        participants = self.directory.snapshot().get(name, ())
        current = [x for x in participants if x.kind == 'agent' and 'request' in x.capabilities
                   and x.network_id == self.config.network_id and x.broker_epoch == (self.config.broker_epoch or '')]
        if not current or len({x.fingerprint for x in current}) != 1:
            raise DefinitionError(f'agent {name} is unavailable or cannot accept requests')
        return current[0]

    async def result(self, task_id: str) -> TaskResult:
        """Wait for the terminal result of a submitted request.

        One journal read per poll; the interval grows from 20 ms to 250 ms, so a
        quick answer is seen at once and a long wait stays cheap.
        """
        delay = RESULT_POLL_SECONDS
        while not self._closed:
            result = await self.store.resolve_submission_expiry(task_id)  # also the terminal task result
            if result is not None:
                return result
            await asyncio.sleep(delay)
            delay = min(delay * RESULT_POLL_GROWTH, RESULT_POLL_MAX_SECONDS)
        raise RuntimeError('Client is closed')

    async def publish(self, topic: str, payload: JSONValue, *, session_id: str | None = None,
                      key: str | None = None, expires_at: datetime | None = None,
                      event_time: datetime | None = None, source_version: int | None = None) -> str:
        """Publish an event to every subscriber of topic and return its event_id.

        The broker ACK is awaited, but the dev broker is not durable: an event
        no subscriber admitted before a broker restart is lost.
        """
        if self._closed:
            raise RuntimeError('Client is closed')
        topic = validate_event_topic(topic)
        now = self.clock.now()
        envelope = Envelope(event_id=str(uuid4()), network_id=self.config.network_id,
            kind=topic, sender=self.sender, recipient=None, topic=topic, task_id=None,
            parent_task_id=None, correlation_id=None, session_id=session_id, created_at=now,
            deadline=now + timedelta(seconds=self.config.limits.deadline_seconds), traceparent=None,
            payload=payload, key=key, event_time=event_time, expires_at=expires_at,
            source_version=source_version)
        await self.bus.publish(envelope)
        return envelope.event_id

    async def event_result(self, event_id: str, agent: str) -> TaskResult:
        """Wait until agent admits and finishes the event published as event_id.

        An event the agent never admits has no task; bound the wait with
        asyncio.timeout when the subscriber may be absent.
        """
        task_id = event_task_id(event_id, validate_name(agent))
        delay = RESULT_POLL_SECONDS
        while not self._closed:
            task = await self.store.get_task(task_id)
            if task and task.status in _TERMINAL:
                return TaskResult(task_id=task.task_id, output=task.result,
                                  responding_agent=task.agent_id, status=task.status, error=task.error)
            await asyncio.sleep(delay)
            delay = min(delay * RESULT_POLL_GROWTH, RESULT_POLL_MAX_SECONDS)
        raise RuntimeError('Client is closed')

    async def events(self, task_id):
        # Subscribe first, then read the journal: the poll bridges publish/read races.
        if await self.store.get_task(task_id) is None and await self.store.get_submission_envelope(task_id) is None:
            raise ValueError('unknown task or submission')
        sub = await self.bus.subscribe([self.bus.control_topic('progress')], 'progress.' + uuid4().hex)
        sequence = 0
        live = None
        try:
            live = asyncio.create_task(anext(sub))
            while not self._closed:
                page = await self.store.read_progress_page(task_id, after_sequence=sequence, limit=64)
                for sequence, event in page:
                    yield event
                if live is not None and live.done():
                    try:
                        delivery = live.result()
                    except (StopAsyncIteration, BrokerUnavailable):
                        await sub.aclose()
                        live = None
                    else:
                        if delivery.envelope.task_id == task_id:
                            await self.store.append_progress(task_id, delivery.envelope)
                        await self.bus.ack(delivery)
                        live = asyncio.create_task(anext(sub))
                if live is None:
                    try:
                        sub = await self.bus.subscribe([self.bus.control_topic('progress')], 'progress.' + uuid4().hex)
                    except BrokerUnavailable:
                        pass
                    else:
                        live = asyncio.create_task(anext(sub))
                task = await self.store.get_task(task_id)
                terminal = task.status in _TERMINAL if task else await self.store.resolve_submission_expiry(task_id) is not None
                if terminal and not page:
                    # The final progress may commit with the result after our last read.
                    if not await self.store.read_progress_page(task_id, after_sequence=sequence, limit=1):
                        break
                    continue
                await asyncio.sleep(.02)
        finally:
            if live:
                live.cancel()
                await asyncio.gather(live, return_exceptions=True)
            await sub.aclose()

    async def close(self) -> None:
        self._closed = True


class AgentClient:
    def __init__(self, client, name):
        self.client, self.name = client, name

    async def submit(self, input: JSONValue, session_id: str | None = None) -> TaskSubmission:
        self.client._target(self.name)
        now = self.client.clock.now()
        envelope = Envelope(event_id=str(uuid4()), network_id=self.client.config.network_id,
            kind='request', sender=self.client.sender, recipient=self.name, topic=None,
            task_id=str(uuid4()), parent_task_id=None, correlation_id=None,
            session_id=str(uuid4()) if session_id is None else session_id, created_at=now,
            deadline=now + timedelta(seconds=self.client.config.limits.deadline_seconds),
            traceparent=None, payload=input)
        return await self.client.store.submit(envelope)

    async def ask(self, input: JSONValue, session_id: str | None = None) -> TaskResult:
        submission = await self.submit(input, session_id)
        return await self.client.result(submission.task_id)


class Conversation:
    """A session answered by one agent at a time; conversation_id is its session_id."""

    def __init__(self, client, conversation_id: str, active_agent: str):
        self.client, self.conversation_id, self.active_agent = client, conversation_id, active_agent

    async def refresh(self) -> str:
        """Read the agent that answers next from the journal."""
        record = await self.client.store.get_conversation(self.conversation_id)
        self.active_agent = record.active_agent
        return self.active_agent

    async def submit(self, input: JSONValue) -> TaskSubmission:
        """Send the next message; a message queued before a handoff reaches the new owner."""
        return await AgentClient(self.client, await self.refresh()).submit(input, self.conversation_id)

    async def ask(self, input: JSONValue) -> TaskResult:
        submission = await self.submit(input)
        result = await self.client.result(submission.task_id)
        await self.refresh()
        return result
