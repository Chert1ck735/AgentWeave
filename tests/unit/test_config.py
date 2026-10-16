"""Config tests catch priority reversal, secret leakage and profile mixing."""
import importlib
from uuid import uuid4
import pytest


def api():
    return importlib.import_module('agentweave.config')


def test_defaults_and_profiles_are_isolated_without_writes(tmp_path):
    c = api()
    with pytest.raises(c.ConfigError, match='identity'):
        c.resolve(tmp_path)
    project_id = str(uuid4())
    dev = c.resolve(tmp_path, overrides={'project_id':project_id, 'network_id':str(uuid4())})
    server = c.resolve(tmp_path, 'server', overrides={'project_id':project_id, 'network_id':str(uuid4())})
    assert dev.root == tmp_path.resolve()
    assert dev.network_id != server.network_id
    assert dev.limits.deadline_seconds == 120
    assert dev.limits.max_handoffs == 8 and dev.limits.max_model_calls == 20
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        c.resolve(tmp_path, 'other')


def test_toml_roundtrip_and_override_priority(tmp_path):
    c = api()
    cfg = c.resolve(tmp_path, overrides={'project_id':str(uuid4()), 'network_id':str(uuid4()), 'broker_url':'127.0.0.1:19092', 'limits':{'max_model_calls':5}})
    path = tmp_path / '.agentweave/dev/config.toml'
    c.write_config(cfg, path)
    loaded = c.resolve(tmp_path)
    assert loaded == cfg
    assert c.resolve(tmp_path, overrides={'limits':{'max_model_calls':7}}).limits.max_model_calls == 7
    assert c.resolve(tmp_path, overrides={'limits':{'max_model_calls':7}}).limits.deadline_seconds == 120
    with pytest.raises(ValueError):
        c.resolve(tmp_path, overrides={'unknown':'value'})
    with pytest.raises(ValueError):
        c.resolve(tmp_path, overrides={'limits':{'max_model_calls':'7'}})


def test_named_config_definition_priority_and_secret_refs(tmp_path, monkeypatch):
    c = api()
    d = importlib.import_module('agentweave.definitions')
    monkeypatch.setenv('MY_KEY', 'do-not-persist')
    (tmp_path / 'agentweave.toml').write_text('''[agents.assistant]
instructions = "Привет\\nWorld"
memory = "stateful"
concurrency = 2
[agents.assistant.model]
provider = "openai"
model_id = "chosen"
api_key_env = "MY_KEY"
''')
    a = d.load_definition('assistant', tmp_path)
    assert a.memory == 'stateful' and a.concurrency == 2
    assert a.model.api_key_env == 'MY_KEY'
    assert c._resolve_agent(a, tmp_path, overrides={'concurrency':3}).concurrency == 3
    assert c.SecretRef(env='MY_KEY').env == 'MY_KEY'
    with pytest.raises(ValueError):
        c.SecretRef(env='MY_KEY', value='secret')
    text = (tmp_path / 'agentweave.toml').read_text()
    assert 'do-not-persist' not in text


def test_unknown_toml_options_fail_before_definition_use(tmp_path):
    c = api()
    (tmp_path / 'agentweave.toml').write_text('[profiles.dev]\nunknown = "bad"\n')
    with pytest.raises(ValueError):
        c.resolve(tmp_path)


def test_project_serializer_roundtrips_env_references_and_escaping(tmp_path, monkeypatch):
    c = api()
    d = importlib.import_module('agentweave.definitions')
    monkeypatch.setenv('MY_KEY', 'never-read-this-secret')
    data = {'project_id':str(uuid4()), 'agents':{'assistant':{'instructions':'quote " slash \\ newline\nПривет',
        'model':{'provider':'openai', 'model_id':'chosen', 'api_key_env':'MY_KEY'}}}}
    path = tmp_path / 'agentweave.toml'
    c.write_config(data, path)
    a = d.load_definition('assistant', tmp_path)
    assert a.instructions == 'quote " slash \\ newline\nПривет'
    assert a.model.api_key_env == 'MY_KEY'
    assert 'never-read-this-secret' not in path.read_text()
    assert path.stat().st_mode & 0o077 == 0


def test_atomic_writer_validation_preserves_existing_file(tmp_path):
    c = api()
    path = tmp_path / 'agentweave.toml'
    path.write_text('existing')
    with pytest.raises(ValueError):
        c.write_config({'agents':{'assistant':{'model':{'provider':'openai', 'model_id':'x', 'options':{'api_key':'bad'}}}}}, path)
    assert path.read_text() == 'existing'
    assert list(tmp_path.iterdir()) == [path]


def test_profile_snapshot_cannot_change_root_or_selected_profile(tmp_path):
    c = api()
    cfg = c.resolve(tmp_path, overrides={'project_id':str(uuid4()), 'network_id':str(uuid4())})
    c.write_config(cfg.model_copy(update={'root':tmp_path/'other'}), tmp_path/'.agentweave/dev/config.toml')
    with pytest.raises(c.ConfigError):
        c.resolve(tmp_path)


def test_configuration_names_cannot_be_ambiguous(tmp_path):
    c = api()
    (tmp_path/'agentweave.toml').write_text('[agents.same]\n[toolboxes.same]\n')
    with pytest.raises(c.ConfigError):
        c._read_project(tmp_path)


def test_writer_rejects_invalid_profile_settings_before_write(tmp_path):
    c = api()
    for settings in [{'network_id':'not-a-uuid'}, {'unknown':1}, {'limits':{'max_handoffs':9}}]:
        with pytest.raises(ValueError):
            c.write_config({'profiles':{'dev':settings}}, tmp_path/'agentweave.toml')
    assert not list(tmp_path.iterdir())


def test_nested_model_override_preserves_definition_provider_and_model(tmp_path):
    c = api()
    a = importlib.import_module('agentweave')
    definition = a.Agent('assistant', model=a.Model.openai('chosen', options={'max_tokens':40}))
    (tmp_path/'agentweave.toml').write_text('[agents.assistant.model.options]\nmax_tokens=50\n')
    resolved = c._resolve_agent(definition, tmp_path, {'model':{'options':{'temperature':0.5}}})
    assert resolved.model.model_id == 'chosen'
    assert resolved.model.api_key_env == 'OPENAI_API_KEY'
    assert resolved.model.options == {'max_tokens':50, 'temperature':0.5}


@pytest.mark.parametrize('options', [
    {'nullable':None, 'kept':1},
    {'nested':{'nullable':None, 'kept':1}},
    {'nested':[{'nullable':None}]},
])
def test_custom_null_options_are_rejected_before_config_replacement(tmp_path, options):
    c = api()
    path = tmp_path/'agentweave.toml'
    path.write_text('[agents.previous]\ninstructions="keep this file"\n')
    before = path.read_bytes()
    data = {'agents':{'assistant':{'model':{'provider':'custom', 'model_id':'chosen',
        'factory_entrypoint':'example:factory', 'options':options}}}}
    with pytest.raises(c.ConfigError, match='represented'):
        c.write_config(data, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_optional_model_field_omission_roundtrips_without_custom_null_dialect(tmp_path):
    c = api()
    d = importlib.import_module('agentweave.definitions')
    models = importlib.import_module('agentweave.models')
    spec = models.ModelSpec(provider='custom', model_id='chosen', factory_entrypoint='example:factory', options={'kept':1})
    values = spec.model_dump(mode='json', exclude_none=True)
    assert 'base_url' not in values and 'api_key_env' not in values
    data = {'agents':{'assistant':{'model':values}}}
    path = tmp_path/'agentweave.toml'
    c.write_config(data, path)
    loaded = d.load_definition('assistant', tmp_path)
    assert loaded.model.options == {'kept':1}
    assert loaded.model.base_url is None and loaded.model.api_key_env is None
    assert loaded.model.factory_entrypoint == 'example:factory'


def test_subscription_policy_toml_roundtrip(tmp_path):
    from agentweave import Agent, SubscriptionPolicy
    from agentweave.config import write_config, _resolve_agent
    from agentweave.definitions import fingerprint
    agent = Agent('counter')
    @agent.on('numbers.received')
    async def handle(ctx, event):
        return event
    original = fingerprint(agent)
    write_config({'agents':{'counter':{'max_inflight':7, 'subscription_policies':{
        'numbers.received':{'queue_limit':3, 'concurrency':2, 'ordering':'unordered', 'overflow':'reject'}}}}}, tmp_path / 'agentweave.toml')
    resolved = _resolve_agent(agent, tmp_path)
    assert resolved.max_inflight == 7
    assert resolved.handlers['numbers.received'].policy == SubscriptionPolicy(queue_limit=3, concurrency=2, ordering='unordered', overflow='reject')
    assert fingerprint(resolved) != original
