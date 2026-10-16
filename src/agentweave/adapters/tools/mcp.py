"""Tools of one MCP server over stdio or Streamable HTTP, through the official SDK's public client.

One background task owns the connection: it connects (the SDK negotiates the
protocol version), lists the tools, then follows the list by change
notifications and, without them, by polling every refresh_seconds. A lost
connection is reopened with fresh schemas. Calls never retry here: the shared
ToolExecutor owns retries and the journal, and the SDK's own resend after a
schema mismatch is bypassed so a changed revision is never called with old
arguments.
"""
import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from ...definitions.tool import ToolSchema, schema_revision
from ...definitions.toolbox import McpToolboxSpec
from ...errors import AgentWeaveError, ToolNotStarted, ToolRequestRejected, ToolRevisionChanged
from ...messages import JSONValue, validate_name

_logger = logging.getLogger('agentweave.mcp')
RECONNECT_SECONDS = (0.2, 10.0)
CONNECT_SECONDS = 30
MAX_LIST_PAGES = 100
STABLE_SECONDS = 30
# JSON-RPC codes that mean the server refused the call without running the tool.
_REFUSALS = frozenset((-32600, -32601, -32602))
_REQUEST_TIMEOUT = -32001


def _sdk_timeout(limit: float) -> float:
    """Slightly below the executor's attempt limit, so the call ends with the SDK's clean timeout."""
    return max(limit * 0.9, limit - 1)


def _call_error(error) -> Exception:
    """Only a definite protocol refusal is a known 'not run'; a lost or late answer may follow an effect."""
    from mcp_types import HEADER_MISMATCH
    if error.code == HEADER_MISMATCH:
        return ToolRevisionChanged('the MCP server reports a changed tool schema')
    if error.code in _REFUSALS:
        return ToolRequestRejected(error.code, f'the MCP server rejected the call (code {error.code})')
    if error.code == _REQUEST_TIMEOUT:
        return TimeoutError('the MCP tool did not answer in time')
    return ConnectionError(f'the MCP call failed (code {error.code})')
_BREAKS = ('\r', '\n', '\x00')


class McpToolError(AgentWeaveError):
    """The MCP tool reported an error; it may have acted before failing."""


class McpTool:
    """One server tool as the ToolExecutor runs it: published schema, policy and live connection."""

    def __init__(self, tool_id: str, remote_name: str, schema: ToolSchema, spec: McpToolboxSpec,
                 connection: 'McpBinding') -> None:
        self.name, self.remote_name, self.connection = tool_id, remote_name, connection
        self.schema = schema.model_copy(update={'tool_id': tool_id})
        self.description, self.input_schema, self.output_schema = schema.description, schema.input_schema, schema.output_schema
        self.retry, self.timeout, self.allowed_callers = spec.retry, spec.timeout, spec.allowed_callers

    async def invoke(self, call, context) -> JSONValue:
        return await self.connection.invoke(self.remote_name, call, revision=self.schema.revision)


class McpBinding:
    def __init__(self, spec: McpToolboxSpec, *, environ: Mapping[str, str] | None = None,
                 cwd: Path | str | None = None, on_change: Callable[[], Awaitable[None]] | None = None) -> None:
        self.spec, self.cwd, self.on_change = spec, cwd, on_change
        self._environ = os.environ if environ is None else environ
        self._schemas: dict[str, ToolSchema] = {}
        self._client = None
        self._runner: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self._changed = asyncio.Event()
        self._closing = False
        self._failure: BaseException | None = None

    def __repr__(self) -> str:
        return f'McpBinding({self.spec.transport} {self.spec.namespace})'

    def headers(self) -> dict[str, str]:
        """HTTP headers from environment references; never from configuration or the server."""
        return {name: self._secret(source.env, f'header {name}') for name, source in self.spec.headers.items()}

    def _secret(self, env: str, purpose: str) -> str:
        value = self._environ.get(env)
        if not value:
            raise ToolNotStarted(f'secret environment variable {env} for {purpose} is not set')
        if any(mark in value for mark in _BREAKS):
            raise ToolNotStarted(f'the secret for {purpose} contains a line break')
        return value

    async def connect(self) -> None:
        """Start the connection task and wait for the first tool listing."""
        if self._runner is None:
            self._runner = asyncio.create_task(self._run(), name='aw-mcp-connection')
        waiting = asyncio.create_task(self._connected.wait())
        try:
            await asyncio.wait([waiting, self._runner], timeout=CONNECT_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiting.cancel()
        if not self._connected.is_set():
            reason = type(self._failure).__name__ if self._failure is not None else 'timeout'
            await self.close()
            raise ToolNotStarted(f'could not connect to the MCP server ({reason})')

    async def list_schemas(self) -> list[ToolSchema]:
        """The server tools currently published, named as on the server."""
        return list(self._schemas.values())

    async def invoke(self, remote_name: str, call, *, revision: str | None = None) -> JSONValue:
        from mcp import MCPError
        client, current = self._client, self._schemas.get(remote_name)
        if client is None or current is None:
            raise ToolNotStarted('the MCP server is not connected or no longer offers this tool')
        if revision is not None and current.revision != revision:
            # The server changed the tool after this call was checked: never send old arguments.
            raise ToolRevisionChanged('tool schema changed; refresh and call again')
        try:
            result = await client.session.call_tool(remote_name, call.arguments,
                                                    read_timeout_seconds=_sdk_timeout(self.spec.timeout))
        except MCPError as error:
            raise _call_error(error) from None
        except TimeoutError:
            raise TimeoutError('the MCP tool did not answer in time') from None
        except Exception as error:
            raise ConnectionError(f'the MCP call failed ({type(error).__name__})') from None
        if result.is_error:
            # Server-written text may echo what it received; it stays in the debug log only.
            text = ' '.join(block.text for block in result.content if getattr(block, 'type', None) == 'text')
            _logger.debug('MCP tool error: %s', text[:500], extra={'tool': remote_name})
            raise McpToolError('the MCP tool reported an error')
        if result.structured_content is not None:
            return result.structured_content
        return [block.model_dump(mode='json', by_alias=True, exclude_none=True) for block in result.content]

    async def close(self) -> None:
        """Close the connection; a stdio server this binding started stops with it."""
        self._closing = True
        self._changed.set()
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

    def _client_for(self, stack: AsyncExitStack):
        from mcp import Client, StdioServerParameters
        if self.spec.transport == 'stdio':
            env = {name: self._secret(source.env, f'environment {name}') for name, source in self.spec.env.items()}
            server = StdioServerParameters(command=self.spec.command, args=list(self.spec.args), env=env or None,
                                           cwd=self.cwd)
        else:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client
            http = httpx2.AsyncClient(headers=self.headers(), trust_env=False, follow_redirects=False,
                                      timeout=httpx2.Timeout(self.spec.timeout, read=max(300, self.spec.timeout)))
            stack.push_async_callback(http.aclose)
            server = streamable_http_client(self.spec.url, http_client=http)
        return Client(server, cache=None, message_handler=self._legacy_notification)

    async def _legacy_notification(self, message) -> None:
        """Older MCP servers announce list changes as plain notifications."""
        if getattr(getattr(message, 'root', message), 'method', None) == 'notifications/tools/list_changed':
            self._changed.set()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        delay = RECONNECT_SECONDS[0]
        while not self._closing:
            connected_at = None
            try:
                async with AsyncExitStack() as stack:
                    client = await stack.enter_async_context(self._client_for(stack))
                    await self._refresh(client)
                    # Calls use the connection only once it has the server's current schemas.
                    self._client, connected_at = client, loop.time()
                    self._connected.set()
                    await self._follow(client, stack)
            except asyncio.CancelledError:
                raise
            except ToolNotStarted as error:
                self._failure = error
                if not self._connected.is_set():
                    return  # a configuration error (e.g. a missing secret) never heals by retrying
                _logger.error('MCP reconnection is not possible: %s', error, extra={'namespace': self.spec.namespace})
            except Exception as error:
                self._failure = error
                _logger.warning('MCP connection lost; reconnecting (%s)', type(error).__name__,
                                extra={'namespace': self.spec.namespace})
            finally:
                self._client = None
            if self._closing:
                return
            if connected_at is not None and loop.time() - connected_at >= STABLE_SECONDS:
                delay = RECONNECT_SECONDS[0]  # only a connection that lasted resets the backoff
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_SECONDS[1])

    async def _follow(self, client, stack: AsyncExitStack) -> None:
        """Refresh on every change event, and at least every refresh_seconds."""
        from mcp.client.subscriptions import ListenNotSupportedError
        try:
            subscription = await stack.enter_async_context(client.listen(tools_list_changed=True))
            events = aiter(subscription)
        except ListenNotSupportedError:
            events = None  # legacy notifications arrive through the message handler
        pending = None
        try:
            while not self._closing:
                changed = asyncio.create_task(self._changed.wait())
                waiters = [changed]
                if events is not None:
                    pending = pending or asyncio.create_task(anext(events))
                    waiters.append(pending)
                await asyncio.wait(waiters, timeout=self.spec.refresh_seconds, return_when=asyncio.FIRST_COMPLETED)
                changed.cancel()
                if pending is not None and pending.done():
                    finished, pending = pending, None
                    if isinstance(finished.exception(), StopAsyncIteration):
                        raise ConnectionError('the MCP change subscription ended')
                    finished.result()
                self._changed.clear()
                await self._refresh(client)
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def _refresh(self, client) -> None:
        tools, cursor = [], None
        async with asyncio.timeout(self.spec.timeout):  # a hung server must not freeze the catalog
            for _ in range(MAX_LIST_PAGES):
                page = await client.list_tools(cursor=cursor, cache_mode='refresh')
                tools.extend(page.tools)
                cursor = page.next_cursor
                if cursor is None:
                    break
        schemas = {}
        for tool in tools:
            if self.spec.allow is not None and tool.name not in self.spec.allow:
                continue
            try:
                validate_name(f'{self.spec.namespace}.{tool.name}')
            except ValueError:
                _logger.warning('skipping an MCP tool whose name is not a logical name', extra={'tool': tool.name[:200]})
                continue
            input_schema = tool.input_schema or {'type': 'object'}
            output_schema = tool.output_schema or {}
            description = tool.description or ''
            schemas[tool.name] = ToolSchema(tool_id=tool.name, description=description, input_schema=input_schema,
                output_schema=output_schema, revision=schema_revision(input_schema, output_schema, description))
        if schemas != self._schemas:
            self._schemas = schemas
            if self.on_change is not None:
                await self.on_change()


def tool_fingerprint(spec: McpToolboxSpec, schema: ToolSchema) -> str:
    """The definition identity of one published MCP tool: its server, name and revision, no secrets."""
    source = [spec.namespace, spec.transport, spec.command, list(spec.args), spec.url, schema.tool_id, schema.revision]
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


__all__ = ['McpBinding', 'McpTool', 'McpToolError', 'tool_fingerprint']
