"""Restricted TOML configuration and explicit resolution, with no bootstrap I/O."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import json
import os
import tempfile
import tomllib
from typing import Literal
from pydantic import Field, field_validator, model_validator, TypeAdapter
from .errors import CapabilityMismatch, ConfigError, DefinitionExists
from .messages import StrictModel, UUIDString, JSONValue, validate_name
from .models import validate_env

class SecretRef(StrictModel):
    env: str

    @field_validator('env')
    @classmethod
    def env_name(cls, value):
        return validate_env(value)

class Limits(StrictModel):
    deadline_seconds: int = Field(default=120, gt=0)
    max_handoffs: int = Field(default=8, ge=0, le=8)
    max_model_calls: int = Field(default=20, gt=0, le=20)
    provider_pool_limit: int = Field(default=4, gt=0)
    heartbeat_seconds: int = Field(default=5, gt=0)
    unavailable_seconds: int = Field(default=15, gt=0)
    envelope_bytes: int = Field(default=1048576, gt=0, le=1048576)

class Retention(StrictModel):
    """How long finished work is kept; unfinished tasks, state, waits and outbox are never expired.

    Dedup IDs live at least replay_window_days + 1 day, and an older event is
    refused instead of being run again.
    """
    terminal_days: int = Field(default=7, gt=0)
    idle_session_days: int = Field(default=30, gt=0)
    replay_window_days: int = Field(default=7, gt=0)


class ProfileConfig(StrictModel):
    profile: Literal['dev', 'server']
    project_id: UUIDString
    network_id: UUIDString
    root: Path
    broker_url: str | None = None
    database_url: str | None = None
    broker_epoch: UUIDString | None = None
    limits: Limits = Limits()
    retention: Retention = Retention()

    @field_validator('root', mode='before')
    @classmethod
    def canonical_root(cls, value):
        if not isinstance(value, (str, Path)):
            raise ValueError('root must be a path')
        return Path(value).resolve()

class _ProjectConfig(StrictModel):
    project_id: UUIDString | None = None
    profiles: dict[str, dict[str, JSONValue]] = {}
    agents: dict[str, dict[str, JSONValue]] = {}
    toolboxes: dict[str, dict[str, JSONValue]] = {}

    @field_validator('profiles')
    @classmethod
    def known_profiles(cls, value):
        if set(value) - {'dev', 'server'}:
            raise ValueError('only dev and server profiles are supported')
        for settings in value.values():
            for name, item in settings.items():
                field = ProfileConfig.model_fields.get(name)
                if field is None:
                    raise ValueError(f'unknown profile setting {name}')
                if name == 'root':
                    ProfileConfig.canonical_root(item)
                else:
                    TypeAdapter(field.rebuild_annotation()).validate_python(item, strict=True)
        return value

    @model_validator(mode='after')
    def unique_definitions(self):
        if set(self.agents) & set(self.toolboxes):
            raise ValueError('agent and toolbox configuration names are ambiguous')
        for name in [*self.agents, *self.toolboxes]:
            validate_name(name)
        return self


def _read_project(root):
    path = Path(root).resolve() / 'agentweave.toml'
    if not path.exists():
        return _ProjectConfig()
    try:
        return _ProjectConfig.model_validate(tomllib.loads(path.read_text(encoding='utf-8')))
    except (ValueError, OSError) as exc:
        raise ConfigError(str(exc)) from exc


def _merge(base, updates):
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), StrictModel):
            model = result[key]
            result[key] = {name:getattr(model, name) for name in type(model).model_fields}
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve(root: Path, profile: str = 'dev', overrides: dict | None = None) -> ProfileConfig:
    root = Path(root).resolve()
    if profile not in ('dev', 'server'):
        raise ConfigError('profile must be dev or server')
    project = _read_project(root)
    data = {'profile':profile, 'root':root, 'limits':{}}
    if project.project_id is not None:
        data['project_id'] = project.project_id
    data = _merge(data, project.profiles.get(profile, {}))
    path = root / '.agentweave' / profile / 'config.toml'
    if path.exists():
        try:
            snapshot = ProfileConfig.model_validate(tomllib.loads(path.read_text(encoding='utf-8')))
        except (ValueError, OSError) as exc:
            raise ConfigError(str(exc)) from exc
        if snapshot.root != root or snapshot.profile != profile:
            raise ConfigError('profile snapshot belongs to another root or profile')
        # Persisted identity and endpoints supply infrastructure data; explicit project settings win.
        data = _merge(snapshot.model_dump(mode='python'), data)
    data = _merge(data, overrides or {})
    if data.get('root') != root or data.get('profile') != profile:
        raise ConfigError('overrides cannot redirect the selected root or profile')
    try:
        return ProfileConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigError(f'invalid or missing profile identity/configuration: {exc}') from exc


def _resolve_agent(definition, root: Path, overrides: dict | None = None):
    """Shared definition priority: defaults < Python < TOML < explicit overrides.

    Resolution is idempotent: an already resolved definition is returned as is
    unless new overrides are given, so a Worker resolving what the CLI resolved
    keeps the CLI overrides and does not add configured hooks a second time.
    """
    from .definitions.agent import Agent
    root = Path(root).resolve()
    if not overrides and definition._resolved_root == root:
        return definition
    project = _read_project(root)
    configured = dict(project.agents.get(definition.name, {}))
    hook_refs = configured.pop('hooks', None)
    data = _merge({name:getattr(definition, name) for name in Agent.model_fields}, configured)
    data = _merge(data, overrides or {})
    data = _agent_values(data)
    try:
        resolved = Agent(**data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    resolved._handlers = definition._handlers.copy()
    for topic, policy in resolved.subscription_policies.items():
        if topic not in resolved._handlers:
            raise ConfigError(f'policy has no registered handler: {topic}')
        resolved._handlers[topic] = resolved._handlers[topic].model_copy(update={'policy':policy})
    resolved._request = definition._request
    resolved._source_hashes = definition._source_hashes.copy()
    resolved._package_versions = definition._package_versions.copy()
    resolved._before_tool = definition._before_tool.copy()
    resolved._after_tool = definition._after_tool.copy()
    if hook_refs is not None:
        _attach_hook_refs(resolved, hook_refs, root)
    else:
        resolved._configured_before_tool = definition._configured_before_tool.copy()
        resolved._configured_after_tool = definition._configured_after_tool.copy()
    resolved._resolved_root = root
    return resolved


def check_model_capabilities(agent) -> None:
    """A model that cannot call tools cannot serve tools or peers: consult and handoff are tool calls."""
    model = agent.model
    if model is None or model.capabilities.tools:
        return
    if agent.tools:
        raise CapabilityMismatch(f'the model of {agent.name} cannot call tools, but the agent declares tools')
    if agent.peers is not None and (agent.peers.mode == 'discover' or agent.peers.names):
        raise CapabilityMismatch(f'the model of {agent.name} cannot call tools, which consult and handoff need')


@contextmanager
def _config_lock(root: Path):
    """Serialize read-modify-write of agentweave.toml between processes."""
    directory = root / '.agentweave'
    directory.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(directory / 'config.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def save_agent(root: Path, name: str, values: dict, *, replace: bool = False):
    """Add a declarative agent to agentweave.toml and return it, validated like an SDK Agent.

    values are the TOML fields of [agents.NAME]; secrets stay environment
    references. Other definitions and profile settings are kept, also when
    several processes save at once.
    """
    from .definitions.agent import Agent
    root = Path(root).resolve()
    try:
        validate_name(name)
    except ValueError as exc:
        raise ConfigError(f'agent name: {exc}') from None
    if not isinstance(values, dict):
        raise ConfigError('an agent definition is a table of settings')
    if values.get('name', name) != name:
        raise ConfigError(f'the definition names another agent than {name}')
    values = {key: value for key, value in values.items() if key != 'name'}
    settings = {key: value for key, value in values.items() if key != 'hooks'}
    try:
        agent = Agent(**_agent_values({'name':name} | settings))
    except ValueError as exc:
        raise ConfigError(f'invalid definition of {name}') from exc
    if 'hooks' in values:
        _attach_hook_refs(agent, values['hooks'], root)
    check_model_capabilities(agent)
    with _config_lock(root):
        project = _read_project(root)
        if name in project.toolboxes:
            raise DefinitionExists(f'{name} already names a toolbox')
        if name in project.agents and not replace:
            raise DefinitionExists(f'{name} is already defined; pass replace to overwrite it')
        data = project.model_dump(mode='json', exclude_none=True)
        data['agents'] = data['agents'] | {name: values}
        write_config(data, root / 'agentweave.toml')
    agent._resolved_root = root  # built from exactly what was written: nothing left to resolve
    return agent


def _attach_hook_refs(agent, refs, root: Path):
    """Append explicit module:object tool hooks declared in agentweave.toml."""
    from .definitions import load_object
    if not isinstance(refs, dict) or set(refs) - {'before_tool', 'after_tool'}:
        raise ConfigError('hooks accepts only before_tool and after_tool lists')
    for kind in ('before_tool', 'after_tool'):
        values = refs.get(kind, [])
        if not isinstance(values, list) or not all(isinstance(ref, str) for ref in values):
            raise ConfigError(f'hooks.{kind} must be a list of module:object references')
        hooks = []
        for ref in values:
            hook, hashes, versions = load_object(ref, root)
            if not callable(hook):
                raise ConfigError(f'{ref}: {kind} hook must be callable')
            hooks.append(hook)
            agent._source_hashes.update(hashes)
            agent._package_versions.update(versions)
        setattr(agent, '_configured_' + kind, hooks)


def _agent_values(data):
    from .models import ModelSpec
    from .definitions.selections import ToolSelection, PeerSelection
    data = dict(data)
    if isinstance(data.get('model'), dict):
        model = dict(data['model'])
        provider = model.get('provider')
        if 'api_key_env' not in model and provider != 'custom':
            model['api_key_env'] = {'openai':'OPENAI_API_KEY', 'compatible':'OPENAI_API_KEY',
                                    'anthropic':'ANTHROPIC_API_KEY', 'gemini':'GEMINI_API_KEY'}.get(provider)
        data['model'] = ModelSpec.model_validate(model)
    if isinstance(data.get('budget'), dict):
        from .models import ModelBudget
        data['budget'] = ModelBudget.model_validate(data['budget'])
    if isinstance(data.get('tools'), dict):
        data['tools'] = ToolSelection.model_validate(data['tools'])
    elif isinstance(data.get('tools'), list):
        from .definitions.toolbox import HttpToolSpec
        # Configured tools need no Python code: each entry is an HTTP tool specification.
        data['tools'] = [HttpToolSpec.model_validate(item) if isinstance(item, dict) else item
                         for item in data['tools']]
    if isinstance(data.get('peers'), dict):
        data['peers'] = PeerSelection.model_validate(data['peers'])
    return data


def _toml_value(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is bool:
        return 'true' if value else 'false'
    if type(value) in (int, float):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_value(x) for x in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{json.dumps(k)} = {_toml_value(v)}' for k, v in value.items()) + '}'
    raise ConfigError('value cannot be represented in the restricted TOML schema')


def _toml_document(data):
    lines = []
    def emit(table, prefix):
        if prefix:
            lines.append('[' + '.'.join(json.dumps(x, ensure_ascii=False) for x in prefix) + ']')
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f'{json.dumps(key, ensure_ascii=False)} = {_toml_value(value)}')
        for key, value in table.items():
            if isinstance(value, dict):
                emit(value, prefix + [key])
    emit(data, [])
    return '\n'.join(lines) + '\n'


def write_config(config, path: Path):
    """Atomically replace a validated profile or restricted project document."""
    if isinstance(config, ProfileConfig):
        data = ProfileConfig.model_validate(config.model_dump(mode='python')).model_dump(mode='json', exclude_none=True)
    elif isinstance(config, dict):
        checked = _ProjectConfig.model_validate(config)
        # Validate declarations and profiles before any filesystem mutation.
        from .definitions.agent import Agent
        from .definitions.toolbox import configured_toolbox
        for name, values in checked.agents.items():
            Agent(**_agent_values({'name':name} | values))
        for name, values in checked.toolboxes.items():
            configured_toolbox(name, values)
        data = checked.model_dump(mode='json', exclude_none=True)
    else:
        raise ConfigError('expected ProfileConfig or a restricted project mapping')
    text = _toml_document(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
