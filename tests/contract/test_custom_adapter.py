"""Custom model adapters plug in without core edits: directly, by registered name, or by entry point."""
import pytest

INSTANCE = '00000000-0000-0000-0000-0000000000e1'


class EchoAdapter:
    """Proposes one tool call, then answers 'custom reply': a complete offline adapter."""

    def __init__(self):
        self.calls = 0

    def capabilities(self):
        from agentweave.models import ModelCapabilities
        return ModelCapabilities()

    def validate_request(self, request):
        from agentweave.providers import check_capabilities
        check_capabilities(self.capabilities(), request)

    async def generate(self, request):
        from agentweave.testing import text_turn, tool_turn
        from agentweave.models import TurnComplete
        self.calls += 1
        turn = tool_turn('ping', {}) if self.calls == 1 else text_turn('custom reply')
        yield TurnComplete(turn=turn)

    async def close(self):
        return None


@pytest.fixture
async def custom_adapter_harness(store, fake_clock, envelope_factory):
    from agentweave import Agent, Model, tool
    from agentweave._runtime.steps import ModelLoop
    from agentweave._runtime.tools import ToolExecutor

    class Harness:
        tool_effects = 0

        async def ask(self, text):
            @tool
            def ping() -> str:
                """Ping."""
                Harness.tool_effects += 1
                return 'pong'
            adapter = EchoAdapter()
            agent = Agent('assistant', model=Model.custom(adapter), tools=[ping])
            event = envelope_factory(payload=text)
            await store.admit(event, 'def-v1')
            lease = await store.claim(event.task_id, INSTANCE, 30)
            task = await store.get_task(event.task_id)
            tools = ToolExecutor(agent, store, instance_id=INSTANCE, clock=fake_clock)
            try:
                return await ModelLoop(agent, store, adapter, instance_id=INSTANCE, clock=fake_clock,
                                       tools=tools).run(task, lease)
            finally:
                tools.close()
    return Harness()


async def test_custom_adapter_needs_no_core_edit(custom_adapter_harness):
    result = await custom_adapter_harness.ask("hello")
    assert result.output == "custom reply"
    assert custom_adapter_harness.tool_effects == 1


def test_a_registered_factory_is_used_by_its_explicit_name():
    from agentweave.models import ModelSpec
    from agentweave.providers import check_available, create_adapter, register_adapter
    seen = []

    def factory(spec, secrets):
        seen.append((spec.model_id, dict(secrets)))
        return EchoAdapter()
    register_adapter('echo-test', factory)
    spec = ModelSpec(provider='custom', model_id='echo-1', factory_entrypoint='echo-test')
    check_available(spec, {})
    assert isinstance(create_adapter(spec, {'K': 'v'}), EchoAdapter)
    assert seen == [('echo-1', {'K': 'v'})]


def test_an_unknown_factory_name_is_a_clear_definition_error():
    from agentweave.errors import DefinitionError
    from agentweave.models import ModelSpec
    from agentweave.providers import check_available
    with pytest.raises(DefinitionError):
        check_available(ModelSpec(provider='custom', model_id='m', factory_entrypoint='nobody-registered-this'), {})


def test_an_installed_entry_point_loads_only_by_its_exact_name(monkeypatch):
    import importlib.metadata
    from agentweave.models import ModelSpec
    from agentweave.providers import create_adapter
    requested = []

    class EntryPoints:
        def select(self, **query):
            requested.append(query)
            if query.get('name') != 'vendor-model':
                return []
            point = type('Point', (), {'name': 'vendor-model', 'load': lambda self: (lambda spec, secrets: EchoAdapter())})()
            return [point]
    monkeypatch.setattr(importlib.metadata, 'entry_points', lambda: EntryPoints())
    adapter = create_adapter(ModelSpec(provider='custom', model_id='m', factory_entrypoint='vendor-model'), {})
    assert isinstance(adapter, EchoAdapter)
    assert requested == [{'group': 'agentweave.models', 'name': 'vendor-model'}]  # a lookup, not a scan


def test_a_custom_factory_from_toml_needs_no_python_wiring(tmp_path):
    from agentweave.definitions import load_definition
    from agentweave.providers import register_adapter
    register_adapter('echo-toml', lambda spec, secrets: EchoAdapter())
    (tmp_path / 'agentweave.toml').write_text('''
[agents.helper]
instructions = "Help."
model = { provider = "custom", model_id = "echo-1", factory_entrypoint = "echo-toml" }
''')
    agent = load_definition('helper', tmp_path)
    assert agent.model.factory_entrypoint == 'echo-toml'


async def test_the_published_contract_helper_accepts_a_correct_adapter_and_rejects_a_broken_one():
    from agentweave.testing import FakeModel, check_model_adapter, text_turn

    class NoTurn(EchoAdapter):
        async def generate(self, request):
            from agentweave.models import TextDelta
            yield TextDelta(text='never completes')
    turn = await check_model_adapter(FakeModel([text_turn('ok'), text_turn('ok')]))
    assert turn.messages[0].content == 'ok'
    with pytest.raises(AssertionError, match='TurnComplete'):
        await check_model_adapter(NoTurn())


def test_the_offline_example_adapter_runs_without_network():
    import asyncio
    import runpy
    from pathlib import Path
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'examples' / 'custom_adapter.py'),
                               run_name='example')
    assert asyncio.run(namespace['demo']()) == 'custom reply'


def test_an_entry_point_name_provided_twice_is_ambiguous(monkeypatch):
    import importlib.metadata
    from agentweave.errors import DefinitionError
    from agentweave.providers import resolve_factory
    point = type('Point', (), {'name': 'dup', 'load': lambda self: (lambda spec, secrets: EchoAdapter())})

    class EntryPoints:
        def select(self, **query):
            return [point(), point()]
    monkeypatch.setattr(importlib.metadata, 'entry_points', lambda: EntryPoints())
    with pytest.raises(DefinitionError, match='more than one'):
        resolve_factory('dup')


def test_a_factory_that_fails_to_load_or_returns_no_adapter_is_a_definition_error():
    from agentweave.errors import DefinitionError
    from agentweave.models import ModelSpec
    from agentweave.providers import create_adapter, register_adapter
    register_adapter('not-an-adapter', lambda spec, secrets: object())
    with pytest.raises(DefinitionError):
        create_adapter(ModelSpec(provider='custom', model_id='m', factory_entrypoint='not-an-adapter'), {})
    with pytest.raises(DefinitionError):
        create_adapter(ModelSpec(provider='custom', model_id='m', factory_entrypoint='json:no_such_factory'), {})
