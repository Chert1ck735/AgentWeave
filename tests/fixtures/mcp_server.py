"""A real MCP server for tests, over stdio or Streamable HTTP, whose tool list changes while it runs.

    python tests/fixtures/mcp_server.py STATE_DIR stdio [--silent]
    python tests/fixtures/mcp_server.py STATE_DIR http PORT [--silent]

STATE_DIR/tools.json maps extra tool names to descriptions; the server adds,
changes and removes them as the file changes and announces each change unless
--silent. Every effect appends a JSON line to STATE_DIR/effects.log.
"""
import json
import sys
from pathlib import Path
import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.subscriptions import InMemorySubscriptionBus
from mcp.shared.subscriptions import ToolsListChanged

STATE = Path(sys.argv[1])
TRANSPORT = sys.argv[2]
SILENT = '--silent' in sys.argv
BUS = InMemorySubscriptionBus()
SERVER = MCPServer('support', subscriptions=BUS, log_level='CRITICAL')


def record(name: str, text: str) -> None:
    with (STATE / 'effects.log').open('a') as stream:
        stream.write(json.dumps([name, text]) + '\n')


@SERVER.tool()
def lookup(query: str) -> str:
    """Look up a support article."""
    record('lookup', query)
    return f'article:{query}'


@SERVER.tool()
def fail(query: str) -> str:
    """Always fails after acting."""
    record('fail', query)
    raise ValueError('the backend broke')


@SERVER.tool()
def big(size: int) -> str:
    """Return a large answer."""
    record('big', str(size))
    return 'x' * size


def dynamic(name: str):
    def tool(text: str) -> str:
        record(name, text)
        return f'{name}:{text}'
    return tool


async def watch() -> None:
    known: dict[str, str] = {}
    while True:
        path = STATE / 'tools.json'
        wanted = json.loads(path.read_text()) if path.exists() else {}
        changed = False
        for name in [name for name in known if name not in wanted]:
            SERVER.remove_tool(name)
            del known[name]
            changed = True
        for name, description in wanted.items():
            if known.get(name) != description:
                if name in known:
                    SERVER.remove_tool(name)
                SERVER.add_tool(dynamic(name), name=name, description=description)
                known[name] = description
                changed = True
        if changed and not SILENT:
            await BUS.publish(ToolsListChanged())
        await anyio.sleep(0.05)


async def main() -> None:
    async with anyio.create_task_group() as group:
        group.start_soon(watch)
        if TRANSPORT == 'stdio':
            await SERVER.run_stdio_async()
        else:
            await SERVER.run_streamable_http_async(port=int(sys.argv[3]))
        group.cancel_scope.cancel()


if __name__ == '__main__':
    anyio.run(main)
