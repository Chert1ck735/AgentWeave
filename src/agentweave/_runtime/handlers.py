"""Bounded pending subscription input and bounded opaque-function execution."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import inspect
import threading
from ..errors import DefinitionError

_CONTROL = frozenset(('request','reply','receipt','progress','directory','error','tool_call','tool_result','consult','handoff'))


def snapshot(event, policy):
    return (policy.replaceable and event.topic is not None and event.recipient is None
            and event.kind not in _CONTROL and event.key is not None and event.source_version is not None)


class SubscriptionQueue:
    def __init__(self, policy, clock, metrics, name):
        self.policy, self.clock, self.metrics, self.name = policy, clock, metrics, name
        self.items = deque()
        self.changed = asyncio.Condition()

    def measure(self):
        age = (self.clock.now() - self.items[0].envelope.created_at).total_seconds() if self.items else 0
        self.metrics.queue(self.name, len(self.items), age)

    async def put(self, delivery):
        async with self.changed:
            event = delivery.envelope
            if self.policy.overflow == 'latest' and snapshot(event, self.policy):
                for old in self.items:
                    previous = old.envelope
                    if snapshot(previous, self.policy) and (previous.sender, previous.topic, previous.key) == (event.sender,event.topic,event.key):
                        if event.source_version <= previous.source_version:
                            return delivery, 'stale_version'
                        self.items.remove(old)
                        self.items.append(delivery)
                        self.measure()
                        self.changed.notify_all()
                        return old, 'coalesced'
            while len(self.items) >= self.policy.queue_limit:
                if self.policy.overflow != 'wait':
                    return delivery, 'rejected'
                await self.changed.wait()
            self.items.append(delivery)
            self.measure()
            self.changed.notify_all()
            return None, None

    async def get(self):
        async with self.changed:
            while not self.items:
                await self.changed.wait()
            delivery = self.items.popleft()
            self.measure()
            self.changed.notify_all()
            return delivery


class HandlerNotStarted(TimeoutError):
    """Freshness or the invocation limit failed before calling opaque user code."""


class HandlerTimeout(TimeoutError):
    """The runtime's invocation limit expired; the effect of user code is unknown.

    A TimeoutError raised by user code itself is an ordinary handler failure.
    """


class _ExecutorOwner:
    def __init__(self, concurrency):
        self.concurrency = concurrency
        self.pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='aw-handler')
        self.references = 0
        self.outstanding = 0
        self.async_tasks = set()


class HandlerExecutor:
    """Process-local ownership stays alive while native work actually remains."""
    _shared = {}
    _shared_lock = threading.RLock()

    def __init__(self, concurrency, *, owner=None):
        self.key = owner if owner is not None else object()
        self.closed = False
        with self._shared_lock:
            state = self._shared.get(self.key)
            if state is not None and state.concurrency != concurrency:
                raise DefinitionError('cannot change executor concurrency while old ownership remains')
            if state is None:
                state = _ExecutorOwner(concurrency)
                self._shared[self.key] = state
            state.references += 1
            self.state = state

    @classmethod
    def _cleanup(cls, key, state):
        if state.references == 0 and state.outstanding == 0 and cls._shared.get(key) is state:
            del cls._shared[key]
            state.pool.shutdown(wait=False)

    @classmethod
    def _completed_native(cls, key, state, future=None):
        # Native Future callbacks run even if the submitting asyncio loop closed.
        with cls._shared_lock:
            state.outstanding -= 1
            cls._cleanup(key, state)

    async def _reserve(self):
        while True:
            with self._shared_lock:
                if self.closed:
                    raise RuntimeError('handler executor is closed')
                if self.state.outstanding < self.state.concurrency:
                    self.state.outstanding += 1
                    return
            await asyncio.sleep(.01)

    def lend(self) -> None:
        """A running handler awaiting a consult frees its place in the pool meanwhile."""
        with self._shared_lock:
            self.state.outstanding -= 1

    async def reclaim(self) -> None:
        """The handler continues: take a place again, waiting if the pool is full."""
        await self._reserve()

    def retake(self) -> None:
        """Count the place again at once: the handler is ending while it was lent."""
        with self._shared_lock:
            self.state.outstanding += 1

    @staticmethod
    def _fresh(context):
        if context is not None:
            try:
                context._assert_fresh()
            except TimeoutError as error:
                raise HandlerNotStarted(str(error)) from error

    @staticmethod
    def _invalidate(context):
        if context is not None:
            context._active = False
            context._lease_valid = False

    async def _wait_owned(self, future, context, *, cancel):
        try:
            # Unlike awaiting user code, cancelling this wait never waits for
            # opaque cancellation handlers. The resource remains accounted.
            await asyncio.wait([future])
        except BaseException:
            self._invalidate(context)
            if cancel:
                future.cancel()
            raise
        return future.result()

    @staticmethod
    async def _await_result(awaitable):
        return await awaitable

    async def _start_async(self, awaitable, context):
        reserved = False
        try:
            await self._reserve()
            reserved = True
            self._fresh(context)
            task = asyncio.create_task(self._await_result(awaitable), name='aw-opaque-handler')
        except BaseException:
            if reserved:
                self._completed_native(self.key, self.state)
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise
        key, state = self.key, self.state
        with self._shared_lock:
            state.async_tasks.add(task)
        def completed(done):
            with self._shared_lock:
                state.async_tasks.discard(done)
            self._completed_native(key, state)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(completed)
        return task

    async def invoke(self, function, context, argument, timeout):
        scope = asyncio.timeout(timeout)
        if context is not None:
            context._invocation_scope = scope  # a consult pauses it while the handler waits
        started = False  # the limit also covers waiting for a slot, before any user code
        try:
            async with scope:
                if inspect.iscoroutinefunction(function):
                    task = await self._start_async(function(context, argument), context)
                    started = True
                    return await self._wait_owned(task, context, cancel=True)
                await self._reserve()
                try:
                    self._fresh(context)
                    native = self.state.pool.submit(function, context, argument)
                except BaseException:
                    self._completed_native(self.key, self.state)
                    raise
                started = True
                key, state = self.key, self.state
                native.add_done_callback(lambda done: self._completed_native(key, state, done))
                future = asyncio.wrap_future(native)
                def completed(done):
                    if not done.cancelled():
                        done.exception()  # consume late thread errors after timeout
                future.add_done_callback(completed)
                result = await self._wait_owned(future, context, cancel=False)
                if inspect.isawaitable(result):
                    task = await self._start_async(result, context)
                    return await self._wait_owned(task, context, cancel=True)
                return result
        except (TimeoutError, asyncio.CancelledError) as error:
            self._invalidate(context)
            # Only the scope's own TimeoutError is translated; a cancellation stays one.
            if isinstance(error, TimeoutError) and scope.expired() and not isinstance(error, HandlerNotStarted):
                if not started:
                    raise HandlerNotStarted('invocation limit expired before the handler started') from error
                raise HandlerTimeout('handler invocation limit expired') from error
            raise

    def close(self):
        with self._shared_lock:
            if not self.closed:
                self.closed = True
                self.state.references -= 1
                self._cleanup(self.key, self.state)
