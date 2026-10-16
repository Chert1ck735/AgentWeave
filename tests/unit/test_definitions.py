"""Declaration tests catch permissive defaults, collisions, and loader drift."""
import importlib
import sys
from pathlib import Path
import pytest


def api():
    return importlib.import_module('agentweave')


def test_declarations_defaults_and_strict_validation():
    a = api().Agent('assistant')
    assert a.memory == 'stateless' and a.concurrency == 1
    assert a.tools is None and a.peers is None
    for kwargs in [dict(memory='other'), dict(concurrency=0), dict(concurrency='2'), dict(unknown=True), dict(name='../bad')]:
        with pytest.raises((ValueError, TypeError)):
            api().Agent(**({'name':'assistant'} | kwargs))
    assert api().Tools.discover().allow is None
    assert api().Tools.discover(allow=[]).allow == ()
    assert api().Peers.allow(['analyst']).names == ('analyst',)


def test_handlers_register_without_executing_and_reject_duplicates():
    a = api().Agent('helper')
    @a.on_request
    async def answer(ctx, request):
        raise AssertionError('must not run during declaration')
    @a.on('numbers.received', retry='safe')
    def event(ctx, value):
        raise AssertionError('must not run during declaration')
    assert a.request_handler.callable is answer
    assert a.handlers['numbers.received'].retry.safe is True
    assert a.request_handler.retry.safe is False
    with pytest.raises(ValueError):
        a.on_request(answer)
    with pytest.raises(ValueError):
        a.on('numbers.received')(event)
    b = api().Agent('other')
    b.on_request(retry='safe')(answer)
    assert b.request_handler.retry.safe


def test_python_tool_derives_schema_and_rejects_alias_collisions():
    @api().tool
    def add(a: float, b: float = 1) -> float:
        """Sum two values."""
        return a + b
    assert add.schema.input_schema['required'] == ['a']
    assert add.schema.input_schema['properties']['a']['type'] == 'number'
    assert add.schema.output_schema['type'] == 'number'
    assert add.retry.safe is False and add.callable(1, 2) == 3
    toolbox = importlib.import_module('agentweave.definitions.toolbox')
    with pytest.raises(ValueError):
        toolbox.ToolboxDefinition(namespace='calc', bindings={'a-b':add, 'a_b':add})


def test_model_factories_keep_secrets_as_refs_and_reject_unknown_options():
    a = api()
    spec = a.Model.openai('chosen-model', api_key_env='MY_KEY', options={'max_tokens':123})
    assert spec.api_key_env == 'MY_KEY' and spec.provider == 'openai'
    for factory in [a.Model.openai, a.Model.anthropic, a.Model.gemini]:
        with pytest.raises(ValueError):
            factory('chosen', options={'api_key':'secret'})
    with pytest.raises(ValueError):
        a.Model.compatible('chosen', base_url='file:///tmp/no')
    with pytest.raises(ValueError):
        a.Model.openai('chosen', api_key_env='not an env name')


def test_loader_canonical_root_relative_imports_and_fingerprint(tmp_path):
    d = importlib.import_module('agentweave.definitions')
    root = tmp_path / 'мой проект'
    root.mkdir()
    (root / 'pkg').mkdir()
    (root / 'pkg/__init__.py').write_text('')
    (root / 'pkg/helper.py').write_text("NAME = 'helper'\n")
    path = root / 'pkg/app.py'
    path.write_text("from .helper import NAME\nfrom agentweave import Agent\na = Agent(NAME, revision='one')\n")
    link = tmp_path / 'link'; link.symlink_to(root, target_is_directory=True)
    before = sys.path.copy()
    a = d.load_definition('pkg.app:a', link)
    assert a.name == 'helper' and sys.path == before
    digest = d.fingerprint(a)
    assert digest == d.fingerprint(d.load_definition('pkg.app:a', root))
    path.write_text("from agentweave import Agent\na = Agent('helper', revision='two')\n")
    assert d.fingerprint(d.load_definition('pkg.app:a', root)) != digest
    with pytest.raises(ValueError):
        d.load_definition('pkg.app:no_such_object', root)
    outside = tmp_path / 'outside.py'; outside.write_text("from agentweave import Agent\na=Agent('outside')")
    (root / 'escape.py').symlink_to(outside)
    with pytest.raises(ValueError):
        d.load_definition('escape:a', root)


def test_custom_adapter_survives_resolution_and_fingerprint_tracks_source(tmp_path):
    a = api()
    models = importlib.import_module('agentweave.models')
    config = importlib.import_module('agentweave.config')
    class Adapter:
        def capabilities(self):
            return models.ModelCapabilities(tools=False, streaming=False)
        def validate_request(self, request):
            pass
        async def generate(self, request):
            yield models.TextDelta(text='hello')
        async def close(self):
            pass
    adapter = Adapter()
    spec = a.Model.custom(adapter, revision='r1')
    original = a.Agent('custom', model=spec)
    assert config._resolve_agent(original, tmp_path).model.custom_adapter is adapter
    assert 'custom_adapter' not in spec.model_dump()
    d = importlib.import_module('agentweave.definitions')
    assert d.fingerprint(original) != d.fingerprint(a.Agent('custom', model=a.Model.custom(adapter, revision='r2')))


def test_python_definition_priority_and_local_tools_are_preserved(tmp_path):
    a = api()
    c = importlib.import_module('agentweave.config')
    @a.tool
    def add(a: int, b: int) -> int:
        return a + b
    original = a.Agent('assistant', memory='stateful', concurrency=2, tools=[add])
    (tmp_path / 'agentweave.toml').write_text('[agents.assistant]\nconcurrency=4\n')
    resolved = c._resolve_agent(original, tmp_path, {'concurrency':6})
    assert resolved.concurrency == 6 and resolved.memory == 'stateful'
    assert resolved.tools[0].callable is add.callable
    assert c._resolve_agent(original, tmp_path).concurrency == 4


def test_http_mcp_specs_validate_secrets_and_transport():
    t = importlib.import_module('agentweave.definitions.toolbox')
    c = importlib.import_module('agentweave.config')
    schema = {'type':'object', 'properties':{'a':{'type':'integer'}}}
    binding = t.HttpToolSpec(name='add', method='POST', url='https://example.test/add', input_schema=schema,
                             headers={'Authorization':c.SecretRef(env='TOKEN')})
    assert binding.headers['Authorization'].env == 'TOKEN'
    with pytest.raises(ValueError):
        t.HttpToolSpec(name='add', method='POST', url='https://example.test', input_schema=schema,
                       headers={'Authorization':t.Literal(value='secret')})
    with pytest.raises(ValueError):
        t.McpToolboxSpec(namespace='x', transport='stdio', command='server', url='https://example.test')
    with pytest.raises(ValueError):
        t.McpToolboxSpec(namespace='x', transport='http', url='file:///tmp/server')


@pytest.mark.parametrize('options', [[], False, {'max_tokens':True}, {'temperature':float('nan')}, {'context_budget':0}])
def test_model_factories_reject_invalid_option_shapes(options):
    with pytest.raises(ValueError):
        api().Model.openai('chosen', options=options)


def test_loader_relative_dependency_changes_and_rejects_escape_before_effect(tmp_path):
    d = importlib.import_module('agentweave.definitions')
    root = tmp_path/'root'; root.mkdir()
    pkg = root/'pkg'; pkg.mkdir()
    (pkg/'__init__.py').write_text('')
    (pkg/'app.py').write_text('from .helper import VALUE\nfrom agentweave import Agent\na=Agent("helper", instructions=VALUE)')
    helper = pkg/'helper.py'; helper.write_text('VALUE="one"')
    first = d.load_definition('pkg.app:a', root)
    helper.write_text('VALUE="two"')
    second = d.load_definition('pkg.app:a', root)
    assert second.instructions == 'two'
    assert d.fingerprint(first) != d.fingerprint(second)
    marker = tmp_path/'executed'
    outside = tmp_path/'outside.py'
    outside.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\nVALUE="bad"')
    helper.unlink(); helper.symlink_to(outside)
    with pytest.raises(ValueError):
        d.load_definition('pkg.app:a', root)
    assert not marker.exists()


def test_loaded_tool_and_toolbox_fingerprints_include_whole_module(tmp_path):
    d = importlib.import_module('agentweave.definitions')
    path = tmp_path/'app.py'
    source = 'from agentweave import tool\nfrom agentweave.definitions import ToolboxDefinition\n@tool\ndef add(a:int)->int:\n    return a+1\nbox=ToolboxDefinition(namespace="calc", bindings={"add":add})\n'
    path.write_text(source)
    fn = d.load_definition('app:add', tmp_path)
    box = d.load_definition('app:box', tmp_path)
    old = (d.fingerprint(fn), d.fingerprint(box))
    path.write_text(source.replace('return a+1','return a+2'))
    assert d.fingerprint(d.load_definition('app:add', tmp_path)) != old[0]
    assert d.fingerprint(d.load_definition('app:box', tmp_path)) != old[1]


def test_direct_tool_fingerprint_tracks_its_module_source(tmp_path):
    import importlib.util
    d = importlib.import_module('agentweave.definitions')
    path = tmp_path/'direct.py'
    path.write_text('from agentweave import tool\n@tool\ndef add(a:int)->int:\n    return a+1\n')
    spec = importlib.util.spec_from_file_location('direct_fingerprint', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        before = d.fingerprint(module.add)
        path.write_text(path.read_text().replace('return a+1','return a+2'))
        assert d.fingerprint(module.add) != before
    finally:
        del sys.modules[spec.name]


def test_custom_factory_references_are_structural_and_require_implementation():
    m = importlib.import_module('agentweave.models')
    spec = m.ModelSpec(provider='custom', model_id='chosen', factory_entrypoint='mypackage:factory', options={'custom':{'level':1}})
    assert spec.options == {'custom':{'level':1}}
    for reference in ['__import__("os")', 'pkg:factory()', ':factory', 'pkg:']:
        with pytest.raises(ValueError):
            m.ModelSpec(provider='custom', model_id='x', factory_entrypoint=reference)
    with pytest.raises(ValueError):
        m.ModelSpec(provider='custom', model_id='x')


def test_tool_declarations_validate_allowed_callers_and_retry_bounds():
    t = importlib.import_module('agentweave.definitions.tool')
    @api().tool(retry=t.RetryPolicy(safe=True, max_attempts=2))
    def add(a:int)->int:
        return a
    assert add.retry.max_attempts == 2
    with pytest.raises(ValueError):
        t.ToolDefinition(name='add', callable=add.callable, schema=add.schema, allowed_callers=('../outside',))
    with pytest.raises(ValueError):
        t.RetryPolicy(min_delay=10, max_delay=1)


def test_project_absolute_and_lazy_imports_are_isolated_and_fresh(tmp_path):
    d = importlib.import_module('agentweave.definitions')
    roots = [tmp_path/'one', tmp_path/'two']
    for root, value in zip(roots, ['one', 'two']):
        root.mkdir()
        (root/'helper.py').write_text(f'VALUE={value!r}')
        (root/'lazy.py').write_text(f'VALUE={value!r}')
        (root/'pkg').mkdir()
        (root/'pkg/__init__.py').write_text('')
        (root/'pkg/util.py').write_text('VALUE="nested"')
        (root/'app.py').write_text('from helper import VALUE\nimport pkg.util\nfrom agentweave import Agent\na=Agent("helper", instructions=VALUE+pkg.util.VALUE)\n@a.on_request\ndef answer(ctx, request):\n    from lazy import VALUE\n    return VALUE\n')
    before = sys.path.copy()
    one = d.load_definition('app:a', roots[0])
    two = d.load_definition('app:a', roots[1])
    assert one.instructions == 'onenested' and two.instructions == 'twonested'
    assert one.request_handler.callable(None, None) == 'one'
    assert two.request_handler.callable(None, None) == 'two'
    assert sys.path == before and 'helper' not in sys.modules and 'lazy' not in sys.modules
    (roots[0]/'helper.py').write_text('VALUE="new"')
    (roots[0]/'lazy.py').write_text('VALUE="new"')
    new = d.load_definition('app:a', roots[0])
    assert new.instructions == 'newnested'
    assert new.request_handler.callable(None, None) == 'new'


def test_lazy_absolute_import_rejects_symlink_before_execution(tmp_path):
    d = importlib.import_module('agentweave.definitions')
    root = tmp_path/'root'; root.mkdir()
    marker = tmp_path/'executed'
    external = tmp_path/'outside.py'
    external.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\nVALUE="bad"')
    (root/'lazy.py').symlink_to(external)
    (root/'app.py').write_text('from agentweave import Agent\na=Agent("helper")\n@a.on_request\ndef answer(ctx, request):\n    from lazy import VALUE\n    return VALUE\n')
    agent = d.load_definition('app:a', root)
    with pytest.raises(ValueError):
        agent.request_handler.callable(None, None)
    assert not marker.exists()


def test_package_reexport_does_not_execute_child_twice(tmp_path):
    d = importlib.import_module('agentweave.definitions')
    pkg = tmp_path/'pkg'; pkg.mkdir()
    (pkg/'__init__.py').write_text('from .app import a\n')
    (pkg/'app.py').write_text('import sys\nfrom agentweave import Agent\npkg=sys.modules[__package__]\npkg.loads=getattr(pkg,"loads",0)+1\na=Agent("once", instructions=str(pkg.loads))\n')
    assert d.load_definition('pkg.app:a', tmp_path).instructions == '1'


def test_installed_definition_fingerprint_tracks_package_version(tmp_path, monkeypatch):
    d = importlib.import_module('agentweave.definitions')
    installed = tmp_path/'installed'; installed.mkdir()
    package = installed/'version_fixture'; package.mkdir()
    (package/'__init__.py').write_text('from agentweave import Agent\na=Agent("installed")')
    metadata = installed/'version_fixture-1.0.dist-info'; metadata.mkdir()
    (metadata/'top_level.txt').write_text('version_fixture\n')
    info = metadata/'METADATA'
    info.write_text('Name: version-fixture\nVersion: 1.0\n')
    project = tmp_path/'project'; project.mkdir()
    monkeypatch.syspath_prepend(str(installed))
    try:
        old = d.fingerprint(d.load_definition('version_fixture:a', project))
        info.write_text('Name: version-fixture\nVersion: 2.0\n')
        assert d.fingerprint(d.load_definition('version_fixture:a', project)) != old
    finally:
        sys.modules.pop('version_fixture', None)


def test_tool_schema_nested_types_resolve_refs_from_document_root():
    from pydantic import BaseModel
    from jsonschema import Draft202012Validator, ValidationError
    class Item(BaseModel):
        value: int
    def consume(item):
        return item.value
    consume.__annotations__ = {'item':list[Item], 'return':int}
    definition = api().tool(consume)
    validator = Draft202012Validator(definition.schema.input_schema)
    validator.validate({'item':[{'value':1}]})
    with pytest.raises(ValidationError):
        validator.validate({'item':[{'value':'invalid'}]})


def test_fingerprint_distinguishes_same_module_custom_adapter_classes(tmp_path):
    d = importlib.import_module('agentweave.definitions')
    (tmp_path/'app.py').write_text('''from agentweave import Agent, Model
from agentweave.models import ModelCapabilities, TextDelta
class First:
    def capabilities(self):
        return ModelCapabilities()
    def validate_request(self, request):
        pass
    async def generate(self, request):
        yield TextDelta(text="first")
    async def close(self):
        pass
class Second(First):
    async def generate(self, request):
        yield TextDelta(text="second")
one = Agent("same", model=Model.custom(First()))
two = Agent("same", model=Model.custom(Second()))
equivalent = Agent("same", model=Model.custom(First()))
''')
    first = d.load_definition('app:one', tmp_path)
    second = d.load_definition('app:two', tmp_path)
    equivalent = d.load_definition('app:equivalent', tmp_path)
    assert first._source_hashes == second._source_hashes
    assert d.fingerprint(first) != d.fingerprint(second)
    assert d.fingerprint(first) == d.fingerprint(equivalent)
    assert d.fingerprint(first) == d.fingerprint(d.load_definition('app:one', tmp_path))


def test_subscription_policy_validation_and_fingerprint():
    from agentweave import Agent, SubscriptionPolicy
    from agentweave.definitions import fingerprint
    import pytest
    for values in ({'queue_limit':0}, {'concurrency':True}, {'ordering':'sorted'}, {'overflow':'latest'}):
        with pytest.raises(ValueError):
            SubscriptionPolicy(**values)
    agent = Agent('counter', max_inflight=2)
    @agent.on('numbers.received', retry='safe', policy=SubscriptionPolicy(overflow='latest', replaceable=True))
    async def count(ctx, event):
        return event
    before = fingerprint(agent)
    altered = Agent('counter', max_inflight=3)
    altered._handlers = agent._handlers.copy()
    assert fingerprint(altered) != before
    assert agent.handlers['numbers.received'].policy.queue_limit == 64


def test_explicit_invalid_subscription_mapping_is_not_defaulted():
    from agentweave import Agent
    import pytest
    for invalid in ([],False,''):
        with pytest.raises(ValueError):
            Agent('counter',subscription_policies=invalid)
