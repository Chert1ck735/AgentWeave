"""agent create/validate: declarative definitions checked by the same contract as SDK objects."""
from pathlib import Path
import sys
from .._infrastructure.processes import InfrastructureError
from .event import parse_json, read_text


def _always(values):
    return True


def _provider_is(name):
    return lambda values: _get(values, ('model', 'provider')) == name


# (field path, prompt, needed when, flag) in the order an interactive create asks.
FIELDS = (
    (('instructions',), 'instructions (empty for none): ', _always, None),
    (('model', 'provider'), 'provider (openai, compatible, anthropic, gemini, custom): ', _always, '--provider'),
    (('model', 'model_id'), 'model id: ', _always, '--model'),
    (('model', 'base_url'), 'base URL of the compatible endpoint: ', _provider_is('compatible'), '--base-url'),
    (('model', 'factory_entrypoint'), 'adapter factory (registered name or module:object): ',
     _provider_is('custom'), '--factory'),
)


def _names(answer):
    return [name.strip() for name in answer.split(',') if name.strip()]


def _tools_answer(answer):
    return {'mode': 'discover'} if answer == '*' else {'mode': 'discover', 'allow': _names(answer)}


def _peers_answer(answer):
    return {'mode': 'discover'} if answer == '*' else {'mode': 'allow', 'names': _names(answer)}


# Offered after the required values; an empty answer or the end of input keeps the default.
OPTIONAL = (
    (('model', 'api_key_env'), 'environment variable holding the model key (empty for the default): ',
     lambda values: _get(values, ('model', 'provider')) != 'custom', str),
    (('memory',), 'memory, stateless or stateful (empty for stateless): ', _always, str),
    (('tools',), 'allowed tools, comma separated, * for all discovered (empty for none): ', _always, _tools_answer),
    (('peers',), 'allowed peers, comma separated, * for all discovered (empty for none): ', _always, _peers_answer),
)


def _get(values, path):
    for part in path:
        if not isinstance(values, dict) or part not in values:
            return None
        values = values[part]
    return values


def _set(values, path, value):
    """A copy of values with path set to value."""
    head, *rest = path
    if not rest:
        return values | {head: value}
    child = values.get(head)
    return values | {head: _set(child if isinstance(child, dict) else {}, rest, value)}


def _merge(base, updates):
    merged = dict(base)
    for key, value in updates.items():
        nested = isinstance(value, dict) and isinstance(merged.get(key), dict)
        merged[key] = _merge(merged[key], value) if nested else value
    return merged


def _json_definition(source):
    values = parse_json(sys.stdin.read() if source == '-' else read_text(source), 'the definition')
    if not isinstance(values, dict):
        raise InfrastructureError('invalid_json', 'the definition must be one JSON object')
    return values


def _present(pairs):
    return {name: value for name, value in pairs if value is not None}


def _flag_values(args):
    """The TOML fields named by command-line flags; flags left out stay out."""
    model = _present((('provider', args.provider), ('model_id', args.model), ('api_key_env', args.api_key_env),
                      ('base_url', args.base_url), ('factory_entrypoint', args.factory)))
    capabilities = {name: False for name, off in (('tools', args.no_tool_calling), ('streaming', args.no_streaming)) if off}
    if capabilities:
        model['capabilities'] = capabilities
    values = _present((('instructions', args.instructions), ('memory', args.memory),
                       ('concurrency', args.concurrency), ('max_inflight', args.max_inflight)))
    if model:
        values['model'] = model
    if (args.tool or args.all_tools) and args.http_tool:
        raise InfrastructureError('invalid_input', 'choose discovered tools or HTTP tools, not both')
    if args.tool or args.all_tools:
        values['tools'] = {'mode': 'discover'} | ({'allow': list(args.tool)} if args.tool else {})
    elif args.http_tool:
        values['tools'] = [parse_json(spec, 'each --http-tool') for spec in args.http_tool]
    if args.peer or args.discover_peers:
        values['peers'] = {'mode': 'discover'} if args.discover_peers else {'mode': 'allow', 'names': list(args.peer)}
    pricing = _present((('revision', args.pricing_revision), ('input_per_million_usd', args.input_price),
                        ('output_per_million_usd', args.output_price)))
    budget = _present((('max_tokens', args.max_tokens), ('max_cost_usd', args.max_cost_usd)))
    if pricing:
        budget['pricing'] = pricing
    if budget:
        values['budget'] = budget
    return values


def _ask(prompt, required=True):
    """One answer from stdin; None at the end of input when the value is optional."""
    sys.stderr.write(prompt)
    sys.stderr.flush()
    line = sys.stdin.readline()
    if not line and required:
        raise InfrastructureError('missing_value', 'input ended before every required value was given')
    return line.rstrip('\n') if line else None


def _complete(values, interactive):
    """Ask for what is missing (interactive) and refuse to go on without a required value."""
    for path, prompt, needed, flag in FIELDS:
        if _get(values, path) is not None or not needed(values):
            continue
        if interactive:
            values = _set(values, path, _ask(prompt))
        if flag is not None and not _get(values, path):
            raise InfrastructureError('missing_value', f'agent create needs {flag} ({path[-1]})')
    for path, prompt, needed, parse in OPTIONAL if interactive else ():
        if _get(values, path) is None and needed(values):
            answer = _ask(prompt, required=False)
            if answer and answer.strip():
                values = _set(values, path, parse(answer.strip()))
    return values


def settings(definition):
    """What validate and create report: the resolved settings, secrets as references only."""
    from ..definitions import fingerprint
    from ..definitions.agent import Agent
    if not isinstance(definition, Agent):
        kind = type(definition).__name__.removesuffix('Definition').removesuffix('Spec').lower()
        name = getattr(definition, 'name', None) or getattr(definition, 'namespace', None)
        return {'valid': True, 'kind': kind, 'name': name, 'fingerprint': fingerprint(definition)}
    model = definition.model
    return {'valid': True, 'kind': 'agent', 'name': definition.name, 'fingerprint': fingerprint(definition),
            'memory': definition.memory, 'concurrency': definition.concurrency, 'max_inflight': definition.max_inflight,
            'model': None if model is None else model.model_dump(mode='json', exclude_none=True, include={
                'provider', 'model_id', 'api_key_env', 'base_url', 'factory_entrypoint', 'capabilities'}),
            'tools': _tools(definition.tools),
            'peers': None if definition.peers is None else definition.peers.model_dump(mode='json'),
            'budget': None if definition.budget is None else definition.budget.model_dump(mode='json', exclude_none=True),
            'subscriptions': {topic: handler.policy.model_dump(mode='json')
                              for topic, handler in definition.handlers.items()},
            'request_handler': definition.request_handler is not None}


def _tools(tools):
    if tools is None:
        return None
    if isinstance(tools, list):
        return [tool.name for tool in tools]
    return tools.model_dump(mode='json')


def create(root, args):
    """Write [agents.NAME] to agentweave.toml once it passes the SDK contract."""
    from ..config import save_agent
    from ..errors import CapabilityMismatch, ConfigError, DefinitionError, DefinitionExists
    from .agent import safe_details
    values = _json_definition(args.definition) if args.definition else {}
    flags = _flag_values(args)
    # A selection given by flags replaces the file's selection instead of merging with it.
    values = {key: value for key, value in values.items() if key not in flags or key not in ('tools', 'peers')}
    values = _complete(_merge(values, flags), interactive=not args.non_interactive)
    try:
        agent = save_agent(root, args.name, values, replace=args.replace)
    except DefinitionExists as exc:
        raise InfrastructureError('definition_exists', str(exc)) from exc
    except CapabilityMismatch as exc:
        raise InfrastructureError('capability_mismatch', str(exc)) from exc
    except (ConfigError, DefinitionError) as exc:
        raise InfrastructureError('definition_invalid', f'invalid definition of {args.name}',
                                  details=safe_details(exc)) from exc
    return {'created': agent.name, 'path': str(Path(root) / 'agentweave.toml')} | settings(agent)


def validate(root, target, overrides):
    """Load, resolve and check a definition as run would, without infrastructure or a model request."""
    from .agent import load_definition
    return settings(load_definition(target, root, overrides))
