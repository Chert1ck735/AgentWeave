"""One SQLite connection owned by one thread; a whole journal operation runs there in one call.

aiosqlite sent every statement to its thread separately, so a transaction held
SQLite's write lock across many event-loop round trips. Here the store's
operation coroutine is driven to completion inside the database thread: its
awaits on this connection complete immediately (they never suspend), so
BEGIN IMMEDIATE ... COMMIT holds the lock only while SQLite works.

Awaited from the event loop instead (outside an operation), each call is
submitted to the same thread, so the connection is never used concurrently.

Writers of one journal also queue on a file lock (flock) before BEGIN IMMEDIATE.
SQLite's own busy handler polls with sleeps that grow to 100 ms, so with many
writing processes a waiter often slept long after the lock was free; the kernel
hands a released flock to the next waiter at once. busy_timeout still covers
processes that do not take this lock.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import sqlite3
import stat
import threading
import time

# How long a writer waits for the journal's file lock before relying on SQLite's busy_timeout alone.
WRITER_LOCK_SECONDS = 5.0
WRITER_POLL_SECONDS = (0.0005, 0.002)


class OperationSuspended(RuntimeError):
    """A journal operation awaited something other than its own connection."""


def drive(coroutine):
    """Run a coroutine that never suspends to completion and return its value."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise OperationSuspended('a journal operation awaited something outside the journal')


class _Result:
    """Rows of a finished statement, fetched in the database thread."""

    def __init__(self, cursor):
        self.rowcount, self.lastrowid = cursor.rowcount, cursor.lastrowid
        self._rows = cursor.fetchall() if cursor.description is not None else []

    async def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    async def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Call:
    """Awaitable (and async context manager) for one connection call."""

    def __init__(self, connection, function):
        self.connection, self.function = connection, function

    def __await__(self):
        if self.connection.in_thread():
            return self.function()
            yield  # a generator that never yields: inside an operation the await completes at once
        return (yield from asyncio.wrap_future(self.connection.submit(self.function)).__await__())

    async def __aenter__(self):
        return await self

    async def __aexit__(self, *exc):
        return False


class ThreadConnection:
    """The store's connection: the awaitable calls aiosqlite offered, executed in one owned thread."""

    def __init__(self, path, *, row_factory=sqlite3.Row):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='aw-journal')
        self._thread_id = None
        self._connection = None
        self._writer_lock = None
        self._path, self._row_factory = path, row_factory

    def in_thread(self) -> bool:
        return threading.get_ident() == self._thread_id

    def submit(self, function):
        return self._executor.submit(function)

    async def open(self):
        def connect():
            self._thread_id = threading.get_ident()
            self._connection = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
            try:
                self._connection.row_factory = self._row_factory
                fd = os.open(str(self._path) + '-writer.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    os.close(fd)
                    raise PermissionError('the journal writer lock must be a regular file owned by this user')
                self._writer_lock = fd
            except BaseException:
                self._connection.close()
                self._connection = None
                raise
        try:
            await asyncio.wrap_future(self.submit(connect))
        except BaseException:
            self._executor.shutdown(wait=False)
            raise
        return self

    def acquire_writer(self) -> bool:
        """In the journal thread: wait for this journal's writers ahead, then write alone.

        False after WRITER_LOCK_SECONDS (a hung holder): the writer then relies on
        SQLite's busy_timeout like a process that does not take this lock.
        """
        deadline, pause = time.monotonic() + WRITER_LOCK_SECONDS, WRITER_POLL_SECONDS[0]
        while True:
            try:
                fcntl.flock(self._writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(pause)
                pause = min(pause * 2, WRITER_POLL_SECONDS[1])

    def release_writer(self) -> None:
        fcntl.flock(self._writer_lock, fcntl.LOCK_UN)

    async def run(self, operation):
        """Run a whole store operation (a coroutine function using this connection) in the thread."""
        return await asyncio.wrap_future(self.submit(lambda: drive(operation())))

    def execute(self, sql, parameters=()):
        return _Call(self, lambda: _Result(self._connection.execute(sql, parameters)))

    def commit(self):
        return _Call(self, lambda: self._connection.commit())

    def rollback(self):
        return _Call(self, lambda: self._connection.rollback())

    async def close(self):
        def close():
            try:
                self._connection.close()
            finally:
                if self._writer_lock is not None:
                    os.close(self._writer_lock)
                    self._writer_lock = None
        try:
            if self._connection is not None:
                await asyncio.wrap_future(self.submit(close))
        finally:
            await asyncio.to_thread(self._executor.shutdown, True)  # never block the event loop


__all__ = ['OperationSuspended', 'ThreadConnection', 'drive']
