"""Pure participant declarations, project-local loading and fingerprints."""
import hashlib
import builtins
import importlib
import importlib.metadata
import importlib.util
import json
import inspect
from pathlib import Path
import sys
from types import ModuleType
from .agent import Agent, SubscriptionPolicy
from .tool import ToolDefinition, ToolSchema, RetryPolicy, tool
from .toolbox import ToolboxDefinition, HttpToolSpec, Literal, Argument, McpToolboxSpec, PythonBinding, configured_toolbox
from .selections import Tools, Peers
from ..errors import DefinitionError


class _ProjectImporter:
    """Module-scoped import mapping retained by lazy local function imports."""
    def __init__(self, root):
        self.root = root
        self.prefix = '_agentweave_project_' + hashlib.sha256(str(root).encode()).hexdigest()[:20]
        self.hashes = {}
        self.modules = {}
        for name in list(sys.modules):
            if name == self.prefix or name.startswith(self.prefix + '.'):
                del sys.modules[name]
        package = ModuleType(self.prefix)
        package.__path__ = [str(root)]
        package.__package__ = self.prefix
        package.__spec__ = importlib.util.spec_from_loader(self.prefix, loader=None, is_package=True)
        self.modules[self.prefix] = package
        sys.modules[self.prefix] = package

    def _exists(self, name):
        path = self.root.joinpath(*name.split('.'))
        return path.with_suffix('.py').exists() or path.is_dir()

    def load(self, fullname):
        if fullname in self.modules:
            return self.modules[fullname]
        relative = fullname[len(self.prefix) + 1:]
        candidate = self.root.joinpath(*relative.split('.'))
        is_package = candidate.is_dir()
        source = candidate / '__init__.py' if is_package else candidate.with_suffix('.py')
        if not source.is_file():
            raise ModuleNotFoundError(f'local module {relative} is missing')
        source = source.resolve()
        if not source.is_relative_to(self.root):
            raise DefinitionError('definition module escapes the canonical project root')
        parent_name, child = fullname.rsplit('.', 1)
        parent = self.load(parent_name)
        if fullname in self.modules:
            return self.modules[fullname]
        spec = importlib.util.spec_from_file_location(fullname, source,
            submodule_search_locations=[str(source.parent)] if is_package else None)
        module = importlib.util.module_from_spec(spec)
        module.__builtins__ = dict(vars(builtins), __import__=self.import_module)
        self.modules[fullname] = module
        sys.modules[fullname] = module
        raw = source.read_bytes()
        self.hashes[str(source.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        try:
            exec(compile(raw, str(source), 'exec'), module.__dict__)
        except BaseException:
            self.modules.pop(fullname, None)
            sys.modules.pop(fullname, None)
            raise
        setattr(parent, child, module)
        return module

    def import_module(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            fullname = importlib.util.resolve_name('.' * level + name, (globals or {}).get('__package__'))
            if fullname != self.prefix and not fullname.startswith(self.prefix + '.'):
                raise ImportError('relative import escapes the project namespace')
        elif self._exists(name.split('.')[0]):
            fullname = self.prefix + '.' + name
        else:
            return builtins.__import__(name, globals, locals, fromlist, level)
        module = self.load(fullname)
        if fromlist:
            for item in fromlist:
                items = getattr(module, '__all__', ()) if item == '*' else (item,)
                for child in items:
                    if not hasattr(module, child) and hasattr(module, '__path__'):
                        candidate = fullname + '.' + child
                        if self._exists(candidate[len(self.prefix) + 1:]):
                            self.load(candidate)
            return module
        # import pkg.submodule binds pkg, matching Python's ordinary import semantics.
        return self.load(self.prefix + '.' + name.split('.')[0])


def _local_module(module_name, root):
    importer = _ProjectImporter(root)
    try:
        module = importer.load(importer.prefix + '.' + module_name)
    except BaseException:
        for name, value in importer.modules.items():
            if sys.modules.get(name) is value:
                del sys.modules[name]
        raise
    # Fingerprint is a snapshot of sources consumed during declaration, not execution.
    return module, importer.hashes.copy()


def load_object(target: str, root: Path) -> tuple[object, dict[str, str], dict[str, str]]:
    """Import module:object from the project root or an installed package.

    Returns the object with the source hashes and package versions consumed,
    which become part of the definition fingerprint.
    """
    root = Path(root).resolve()
    hashes, versions = {}, {}
    parts = target.split(':') if isinstance(target, str) else ()
    if len(parts) != 2 or not all(p and all(x.isidentifier() for x in p.split('.')) for p in parts):
        raise DefinitionError('target must be module:object or a configured name')
    module_name, object_name = parts
    local_path = root.joinpath(*module_name.split('.'))
    try:
        if local_path.with_suffix('.py').exists() or local_path.is_dir():
            module, hashes = _local_module(module_name, root)
        else:
            module = importlib.import_module(module_name)
            if module.__file__:
                path = Path(module.__file__).resolve()
                hashes[module_name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for dist in importlib.metadata.packages_distributions().get(module_name.split('.')[0], []):
                versions[dist] = importlib.metadata.version(dist)
        obj = module
        for name in object_name.split('.'):
            obj = getattr(obj, name)
    except (ImportError, AttributeError, OSError) as exc:
        raise DefinitionError(f'cannot load {target}: {exc}') from exc
    return obj, hashes, versions


def load_definition(target: str, root: Path) -> Agent | ToolDefinition | ToolboxDefinition:
    root = Path(root).resolve()
    from ..config import _read_project, _agent_values, _resolve_agent
    hashes, versions = {}, {}
    if ':' not in target:
        config = _read_project(root)
        if target in config.agents:
            obj = Agent(**_agent_values({'name':target} | config.agents[target]))
        elif target in config.toolboxes:
            obj = configured_toolbox(target, config.toolboxes[target])
        else:
            raise DefinitionError(f'no configured definition {target}')
    else:
        obj, hashes, versions = load_object(target, root)
    if not isinstance(obj, (Agent, ToolDefinition, ToolboxDefinition, McpToolboxSpec)):
        raise DefinitionError('target must be an Agent, ToolDefinition, ToolboxDefinition or McpToolboxSpec')
    if isinstance(obj, Agent):
        obj._source_hashes = hashes
        obj._package_versions = versions
        obj = _resolve_agent(obj, root)
    else:
        # Pydantic private attrs allow metadata without becoming wire/config fields.
        obj.__pydantic_private__ = dict(obj.__pydantic_private__ or {}, _source_hashes=hashes, _package_versions=versions)
    return obj


def _portable(value):
    if callable(value):
        module = value.__module__
        if module.startswith('_agentweave_project_'):
            module = module.split('.', 1)[1]
        return {'module':module, 'qualname':value.__qualname__}
    if isinstance(value, dict):
        return {key:_portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(x) for x in value]
    return value


def fingerprint(definition) -> str:
    if not isinstance(definition, (Agent, ToolDefinition, ToolboxDefinition, HttpToolSpec, McpToolboxSpec)):
        raise DefinitionError('unsupported definition')
    private = definition.__pydantic_private__ or {}
    data = {'schema_version':1, 'sdk_version':'0.1.0',
            'definition':_portable(definition.model_dump(mode='python')),
            'modules':dict(private.get('_source_hashes', {})), 'packages':dict(private.get('_package_versions', {}))}
    if not data['modules']:
        callables = []
        if isinstance(definition, ToolDefinition):
            callables.append(definition.callable)
        elif isinstance(definition, ToolboxDefinition):
            callables.extend(binding.callable for binding in definition.bindings.values() if isinstance(binding, ToolDefinition))
        elif isinstance(definition, Agent):  # an HTTP spec has no code, only its configuration
            callables.extend(handler.callable for handler in definition.handlers.values())
            if definition.request_handler:
                callables.append(definition.request_handler.callable)
            callables.extend(definition.before_tool_hooks + definition.after_tool_hooks)
            if isinstance(definition.tools, list):
                callables.extend(binding.callable for binding in definition.tools if isinstance(binding, ToolDefinition))
            if definition.model and definition.model.custom_adapter is not None:
                callables.append(type(definition.model.custom_adapter))
        for fn in callables:
            module = inspect.getmodule(fn)
            path = getattr(module, '__file__', None)
            if path and Path(path).is_file():
                data['modules'][module.__name__] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if module:
                for dist in importlib.metadata.packages_distributions().get(module.__name__.split('.')[0], []):
                    data['packages'][dist] = importlib.metadata.version(dist)
    if isinstance(definition, Agent):
        if definition.model and definition.model.custom_adapter is not None:
            data['adapter_implementation'] = _portable(type(definition.model.custom_adapter))
        data['handlers'] = {name:_portable(handler.model_dump(mode='python')) for name, handler in definition.handlers.items()}
        data['request'] = _portable(definition.request_handler.model_dump(mode='python')) if definition.request_handler else None
        data['tool_hooks'] = {'before': _portable(list(definition.before_tool_hooks)),
                              'after': _portable(list(definition.after_tool_hooks))}
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()

__all__ = ['SubscriptionPolicy', 'Agent', 'Tools', 'Peers', 'tool', 'ToolDefinition', 'ToolSchema', 'RetryPolicy',
           'ToolboxDefinition', 'HttpToolSpec', 'Literal', 'Argument', 'McpToolboxSpec', 'PythonBinding', 'load_definition', 'load_object', 'fingerprint']
