"""The model adapter contract and lazy creation of built-in adapters.

Adapters translate a portable ModelRequest into a provider API and stream
ModelEvents back. They never execute tools, retry, or manage budgets: the
runtime owns every one of those decisions. Provider SDKs are imported only
when an adapter is created, so importing agentweave stays light.
"""
from collections.abc import AsyncIterator, Callable, Mapping
import importlib.util
from typing import Protocol, runtime_checkable
from .errors import ConfigError, DefinitionError, ModelError
from .models import ModelCapabilities, ModelEvent, ModelRequest, ModelSpec


@runtime_checkable
class ModelAdapter(Protocol):
    def capabilities(self) -> ModelCapabilities: ...
    def validate_request(self, request: ModelRequest) -> None: ...
    def generate(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...
    async def close(self) -> None: ...


def check_capabilities(capabilities: ModelCapabilities, request: ModelRequest) -> None:
    """Reject a request the model cannot serve instead of silently degrading it."""
    if request.tools and not capabilities.tools:
        raise ModelError('unsupported', 'this model does not support tools')


def pool_name(spec: ModelSpec) -> str:
    """Requests to one provider endpoint share a profile-wide concurrency pool."""
    return f'{spec.provider}:{spec.base_url or "default"}'


AdapterFactory = Callable[[ModelSpec, Mapping[str, str]], ModelAdapter]
_FACTORIES: dict[str, AdapterFactory] = {}


def register_adapter(name: str, factory: AdapterFactory) -> None:
    """Make factory available to custom models that name it as factory_entrypoint."""
    if not isinstance(name, str) or not name or not callable(factory):
        raise DefinitionError('register_adapter needs a name and a callable factory')
    _FACTORIES[name] = factory


def resolve_factory(name: str) -> AdapterFactory:
    """A registered factory, else the installed entry point of exactly that name; nothing is scanned."""
    if name in _FACTORIES:
        return _FACTORIES[name]
    if ':' in name:
        import importlib
        module, attribute = name.split(':')
        try:
            factory = importlib.import_module(module)
            for part in attribute.split('.'):
                factory = getattr(factory, part)
        except Exception as error:
            raise DefinitionError(f'cannot load the model adapter factory {name}') from error
        return factory
    import importlib.metadata
    points = list(importlib.metadata.entry_points().select(group='agentweave.models', name=name))
    if not points:
        raise DefinitionError(f'no model adapter factory named {name} is registered or installed')
    if len(points) > 1:
        raise DefinitionError(f'more than one installed package provides the model adapter {name}')
    try:
        return points[0].load()
    except Exception as error:
        raise DefinitionError(f'cannot load the model adapter factory {name}') from error


def check_available(spec: ModelSpec, secrets: Mapping[str, str]) -> None:
    """Preflight without side effects: provider support, key reference and installed SDK.

    It builds no client and opens nothing, so it is safe in sync and async code;
    only a custom module:object factory reference imports that (trusted) module.
    """
    if spec.provider == 'custom':
        if spec.custom_adapter is None:
            resolve_factory(spec.factory_entrypoint)
        return
    if spec.provider in ('anthropic', 'gemini'):
        raise DefinitionError(f'the native {spec.provider} adapter is planned for v0.2; use '
                              'Model.compatible with the provider\'s OpenAI-compatible endpoint')
    if spec.provider == 'compatible' and not spec.capabilities.streaming:
        raise ConfigError('the compatible adapter streams; an endpoint without streaming is not supported')
    if not secrets.get(spec.api_key_env or ''):
        raise ConfigError(f'environment variable {spec.api_key_env} is not set')
    if importlib.util.find_spec('openai') is None:
        raise ConfigError('the OpenAI integration is not installed: pip install "agentweave[openai]"')


def create_adapter(spec: ModelSpec, secrets: Mapping[str, str]) -> ModelAdapter:
    """Build the adapter for spec; secrets maps environment variable names to values."""
    if spec.provider == 'custom':
        if spec.custom_adapter is not None:
            return spec.custom_adapter
        adapter = resolve_factory(spec.factory_entrypoint)(spec, secrets)  # one lookup, by its exact name
        if not all(callable(getattr(adapter, method, None)) for method in ('capabilities', 'validate_request', 'generate', 'close')):
            raise DefinitionError(f'the factory {spec.factory_entrypoint} did not return a model adapter')
        return adapter
    check_available(spec, secrets)
    if spec.provider == 'compatible':
        from .adapters.models.compatible import CompatibleAdapter
        return CompatibleAdapter(spec, secrets[spec.api_key_env])
    from .adapters.models.openai import OpenAIAdapter
    return OpenAIAdapter(spec, secrets[spec.api_key_env])


__all__ = ['AdapterFactory', 'ModelAdapter', 'check_available', 'check_capabilities', 'create_adapter', 'pool_name',
           'register_adapter', 'resolve_factory']
