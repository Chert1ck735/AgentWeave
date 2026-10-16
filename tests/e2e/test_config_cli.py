"""agent create/validate: TOML definitions equal to SDK ones, checked before any process starts."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import pytest
from tests.e2e.test_dev_cli import ENV, REPO

SECRET = 'API_SECRET_VALUE'


class ConfigCli:
    def __init__(self, root, env):
        self.root, self.env = root, env

    def run(self, *args, input=None):
        return subprocess.run([sys.executable, '-m', 'agentweave', '--root', str(self.root), '--json', *args],
                              input=input, cwd=REPO, env=self.env, capture_output=True, text=True, timeout=60)

    def create(self, name, *, provider='openai', model='gpt-test', memory=None, extra=()):
        args = ['agent', 'create', name, '--non-interactive', '--provider', provider, '--model', model, *extra]
        if memory is not None:
            args += ['--memory', memory]
        result = self.run(*args)
        assert result.returncode == 0, (result.stdout, result.stderr)
        return json.loads(result.stdout)

    def load(self, name):
        from agentweave.definitions import load_definition
        return load_definition(name, self.root)

    @property
    def toml_text(self):
        return (self.root / 'agentweave.toml').read_text()


@pytest.fixture
def config_cli(tmp_path):
    root = tmp_path / 'config project'
    root.mkdir()
    return ConfigCli(root, ENV | {'OPENAI_API_KEY': SECRET})


def failure(result, code, exit_code=2):
    assert result.returncode == exit_code, (result.returncode, result.stdout, result.stderr)
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload['error_code'] == code, payload
    return payload


def test_cli_memory_choice_matches_sdk(config_cli):
    config_cli.create("assistant", memory="stateful")
    assert config_cli.load("assistant").memory == "stateful"
    assert "API_SECRET_VALUE" not in config_cli.toml_text


def test_a_created_definition_is_the_same_agent_as_its_sdk_declaration(config_cli):
    from agentweave import Agent, Model, Peers, Tools
    from agentweave.definitions import fingerprint
    from agentweave.models import ModelBudget, ModelPricing
    config_cli.create('analyst', extra=['--instructions', 'Answer with numbers.', '--tool', 'weather.*',
                                        '--tool', 'calc', '--peer', 'writer', '--max-inflight', '5',
                                        '--concurrency', '2', '--max-tokens', '1000', '--max-cost-usd', '0.10',
                                        '--pricing-revision', 'r1', '--input-price', '0.15', '--output-price', '0.60'])
    sdk = Agent('analyst', instructions='Answer with numbers.', model=Model.openai('gpt-test'),
                tools=Tools.discover(allow=['weather.*', 'calc']), peers=Peers.allow(['writer']),
                max_inflight=5, concurrency=2,
                budget=ModelBudget(max_tokens=1000, max_cost_usd='0.10', pricing=ModelPricing(
                    revision='r1', input_per_million_usd='0.15', output_per_million_usd='0.60')))
    loaded = config_cli.load('analyst')
    assert loaded.model_dump() == sdk.model_dump()
    assert fingerprint(loaded) == fingerprint(sdk)
    assert '"max_cost_usd" = "0.10"' in config_cli.toml_text  # money stays a decimal string, never a float


def test_non_interactive_create_requires_every_mandatory_value(config_cli):
    result = config_cli.run('agent', 'create', 'assistant', '--non-interactive', '--provider', 'openai')
    assert 'model' in failure(result, 'missing_value')['message']
    failure(config_cli.run('agent', 'create', 'assistant', '--non-interactive', '--provider', 'compatible',
                           '--model', 'local'), 'missing_value')  # a compatible endpoint needs its base URL
    failure(config_cli.run('agent', 'create', 'assistant', '--non-interactive', '--provider', 'openai',
                           '--model', 'gpt-test', '--api-key-env', 'sk-not a name'), 'definition_invalid')
    assert not (config_cli.root / 'agentweave.toml').exists()


def test_interactive_create_prompts_only_for_missing_values(config_cli):
    result = config_cli.run('agent', 'create', 'helper', '--memory', 'stateful',
                            input='Be brief.\ncompatible\nlocal-model\nhttp://llm.local:8080/v1\n')
    assert result.returncode == 0, (result.stdout, result.stderr)
    agent = config_cli.load('helper')
    assert (agent.instructions, agent.memory) == ('Be brief.', 'stateful')
    assert (agent.model.provider, agent.model.model_id, agent.model.base_url) == (
        'compatible', 'local-model', 'http://llm.local:8080/v1')
    assert 'memory' not in result.stderr.lower()  # given on the command line: never asked
    assert 'tools' in result.stderr.lower()  # optional values are offered; end of input keeps their defaults


def test_interactive_create_asks_for_the_key_reference_tools_and_peers(config_cli):
    result = config_cli.run('agent', 'create', 'scout', '--provider', 'openai', '--model', 'gpt-test',
                            input='Look around.\nSCOUT_KEY\nstateful\nweather.*, calc\n*\n')
    assert result.returncode == 0, (result.stdout, result.stderr)
    agent = config_cli.load('scout')
    assert (agent.model.api_key_env, agent.memory) == ('SCOUT_KEY', 'stateful')
    assert agent.tools.allow == ('weather.*', 'calc') and agent.peers.mode == 'discover'


def test_a_json_definition_from_stdin_uses_the_same_schema(config_cli):
    definition = {'instructions': 'Hi.', 'memory': 'stateful', 'model': {'provider': 'openai', 'model_id': 'gpt-test'}}
    result = config_cli.run('agent', 'create', 'greeter', '--definition', '-', '--non-interactive',
                            input=json.dumps(definition | {'colour': 'red'}))
    assert any('colour' in str(item['loc']) for item in failure(result, 'definition_invalid')['details'])
    failure(config_cli.run('agent', 'create', 'greeter', '--definition', '-', input='{not json'), 'invalid_json')
    result = config_cli.run('agent', 'create', 'greeter', '--definition', '-', '--non-interactive',
                            input=json.dumps(definition))
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert config_cli.load('greeter').memory == 'stateful'


def test_an_existing_definition_is_replaced_only_on_request(config_cli):
    config_cli.create('assistant')
    failure(config_cli.run('agent', 'create', 'assistant', '--non-interactive', '--provider', 'openai',
                           '--model', 'other'), 'definition_exists')
    assert config_cli.load('assistant').model.model_id == 'gpt-test'
    config_cli.create('assistant', model='other', extra=['--replace'])
    assert config_cli.load('assistant').model.model_id == 'other'


def test_concurrent_creates_keep_every_definition_and_the_profile_settings(config_cli):
    from agentweave.config import _read_project, write_config
    write_config({'profiles': {'dev': {'limits': {'provider_pool_limit': 2}}}}, config_cli.root / 'agentweave.toml')
    names = [f'agent-{index}' for index in range(6)]
    with ThreadPoolExecutor(len(names)) as pool:
        list(pool.map(config_cli.create, names))
    project = _read_project(config_cli.root)
    assert sorted(project.agents) == names
    assert project.profiles['dev']['limits'] == {'provider_pool_limit': 2}


def test_validate_reports_a_capability_mismatch_before_any_infrastructure(config_cli):
    (config_cli.root / 'agentweave.toml').write_text('''
[agents.planner]
model = { provider = "compatible", model_id = "local", base_url = "http://llm.local/v1", capabilities = { tools = false } }
tools = { mode = "discover" }
''')
    payload = failure(config_cli.run('agent', 'validate', 'planner'), 'capability_mismatch')
    assert 'tool' in payload['message']
    failure(config_cli.run('agent', 'create', 'solo', '--non-interactive', '--provider', 'compatible',
                           '--model', 'local', '--base-url', 'http://llm.local/v1', '--no-tool-calling',
                           '--peer', 'planner'), 'capability_mismatch')
    assert not (config_cli.root / '.agentweave' / 'dev').exists()


def test_validate_applies_override_over_toml_over_python_over_defaults(config_cli):
    (config_cli.root / 'app.py').write_text('''from agentweave import Agent
helper = Agent('helper', concurrency=2)
@helper.on_request
async def answer(ctx, request):
    return request.input
''')
    (config_cli.root / 'agentweave.toml').write_text('[agents.helper]\nmemory = "stateful"\n')
    result = config_cli.run('agent', 'validate', 'app:helper')
    assert result.returncode == 0, (result.stdout, result.stderr)
    settings = json.loads(result.stdout)
    assert (settings['valid'], settings['name']) == (True, 'helper')
    assert (settings['memory'], settings['concurrency'], settings['max_inflight']) == ('stateful', 2, 64)
    overridden = json.loads(config_cli.run('agent', 'validate', 'app:helper', '--memory', 'stateless',
                                           '--concurrency', '4').stdout)
    assert (overridden['memory'], overridden['concurrency']) == ('stateless', 4)
    assert overridden['fingerprint'] != settings['fingerprint']


@pytest.mark.parametrize('section, detail', [
    ('[agents.helper.subscription_policies."orders.created"]\noverflow = "latest"\n', 'replaceable'),
    ('[agents.helper.budget]\nmax_cost_usd = "1.00"\n', 'pricing')])
def test_validate_rejects_latest_without_replaceable_and_cost_without_tariff(config_cli, section, detail):
    (config_cli.root / 'app.py').write_text('''from agentweave import Agent
helper = Agent('helper')
@helper.on('orders.created')
async def created(ctx, event):
    return event
''')
    (config_cli.root / 'agentweave.toml').write_text(section)
    payload = failure(config_cli.run('agent', 'validate', 'app:helper'), 'definition_invalid')
    assert detail in json.dumps(payload)


def test_validate_checks_the_key_reference_without_a_model_request(config_cli):
    config_cli.create('assistant', provider='compatible', extra=['--base-url', 'http://127.0.0.1:9/v1'])
    valid = config_cli.run('agent', 'validate', 'assistant')
    assert valid.returncode == 0, (valid.stdout, valid.stderr)  # port 9 discards: any request would fail
    payload = json.loads(valid.stdout)
    assert payload['model'] == {'provider': 'compatible', 'model_id': 'gpt-test', 'api_key_env': 'OPENAI_API_KEY',
                                'base_url': 'http://127.0.0.1:9/v1',
                                'capabilities': {'tools': True, 'streaming': True}}
    assert SECRET not in valid.stdout
    config_cli.env = {key: value for key, value in config_cli.env.items() if key != 'OPENAI_API_KEY'}
    failure(config_cli.run('agent', 'validate', 'assistant'), 'model_key_missing')


def test_validate_of_an_unknown_definition_is_an_input_error(config_cli):
    failure(config_cli.run('agent', 'validate', 'nobody'), 'definition_invalid')
    failure(config_cli.run('agent', 'validate', 'app:missing'), 'definition_invalid')


def test_unreadable_definitions_and_a_conflicting_name_are_input_errors(config_cli, tmp_path):
    binary = tmp_path / 'definition.json'
    binary.write_bytes(b'\xff\xfe{}')
    failure(config_cli.run('agent', 'create', 'a', '--definition', str(binary), '--non-interactive'), 'invalid_input')
    failure(config_cli.run('agent', 'create', 'a', '--definition', '-', '--non-interactive', input='[' * 100000),
            'invalid_json')
    named = json.dumps({'name': 'other', 'model': {'provider': 'openai', 'model_id': 'gpt-test'}})
    failure(config_cli.run('agent', 'create', 'a', '--definition', '-', '--non-interactive', input=named),
            'definition_invalid')
    assert not (config_cli.root / 'agentweave.toml').exists()


def test_tool_and_peer_flags_replace_the_selection_of_a_definition_file(config_cli, tmp_path):
    source = tmp_path / 'definition.json'
    source.write_text(json.dumps({'model': {'provider': 'openai', 'model_id': 'gpt-test'},
                                  'tools': {'mode': 'discover', 'allow': ['calc']},
                                  'peers': {'mode': 'allow', 'names': ['writer']}}))
    config_cli.run('agent', 'create', 'a', '--definition', str(source), '--non-interactive', '--all-tools',
                   '--discover-peers')
    agent = config_cli.load('a')
    assert agent.tools.allow is None and agent.peers.mode == 'discover'
