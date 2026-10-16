"""A model adapter of your own, plugged in without touching AgentWeave.

    python examples/custom_adapter.py

runs offline: the adapter below proposes a tool call, then answers. Use the
same class with Model.custom(...) in an Agent, or register a factory with
agentweave.providers.register_adapter("my-model", factory) and name it in
agentweave.toml as model = { provider = "custom", model_id = "...",
factory_entrypoint = "my-model" }. agentweave.testing.check_model_adapter
checks the adapter contract in your own tests.
"""
import asyncio
from agentweave.models import ModelCapabilities, TextDelta, TurnComplete
from agentweave.providers import check_capabilities
from agentweave.testing import check_model_adapter, text_turn, tool_turn


class ScriptedAdapter:
    """Answers from a fixed script; a real adapter would call its provider here."""

    def __init__(self):
        self.turns = 0

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(tools=True, streaming=True)

    def validate_request(self, request) -> None:
        check_capabilities(self.capabilities(), request)

    async def generate(self, request):
        self.turns += 1
        if self.turns == 1 and request.tools:
            yield TurnComplete(turn=tool_turn(request.tools[0].tool_id, {}))
            return
        yield TextDelta(text='custom ')
        yield TurnComplete(turn=text_turn('custom reply'))

    async def close(self) -> None:
        return None


async def demo() -> str:
    turn = await check_model_adapter(ScriptedAdapter())
    return turn.messages[0].content


if __name__ == '__main__':
    print(asyncio.run(demo()))
