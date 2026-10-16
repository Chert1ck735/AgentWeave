"""Observe real package imports and enforce current/future dependency boundaries."""
import subprocess
import sys
import os
from pathlib import Path


def test_actual_imports_obey_layer_boundaries():
    script = r'''
import builtins, importlib, pkgutil, sys, ast
from pathlib import Path
edges = set()
original = builtins.__import__
def traced(name, globals=None, locals=None, fromlist=(), level=0):
    source = (globals or {}).get('__name__', '')
    package = (globals or {}).get('__package__', '')
    target = importlib.util.resolve_name('.' * level + name, package) if level else name
    if source.startswith('agentweave'):
        edges.add((source, target))
    result = original(name, globals, locals, fromlist, level)
    if source.startswith('agentweave'):
        for item in fromlist or ():
            if item != '*':
                edges.add((source, target + '.' + item))
    return result
builtins.__import__ = traced
original_import_module = importlib.import_module
def traced_import_module(name, package=None):
    source = sys._getframe(1).f_globals.get('__name__', '')
    target = importlib.util.resolve_name(name, package) if name.startswith('.') else name
    if source.startswith('agentweave'):
        edges.add((source, target))
    return original_import_module(name, package)
importlib.import_module = traced_import_module
import agentweave
# Explicit imports cover unloaded internal modules too, as future tasks add them.
for module in pkgutil.walk_packages(agentweave.__path__, agentweave.__name__ + '.'):
    loaded = importlib.import_module(module.name)
    # Static import nodes supplement real imports for uncalled functions and TYPE_CHECKING branches.
    file = getattr(loaded, '__file__', None)
    if file and file.endswith('.py'):
        package = loaded.__package__
        for node in ast.walk(ast.parse(Path(file).read_text())):
            if isinstance(node, ast.Import):
                edges.update((module.name, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                if target:
                    edges.add((module.name, target))
                    edges.update((module.name, target + '.' + alias.name) for alias in node.names if alias.name != '*')
common = ('agentweave.messages', 'agentweave.models', 'agentweave._records', 'agentweave.errors', 'agentweave.definitions')
native = ('openai', 'anthropic', 'google.genai', 'mcp', 'psycopg', 'confluent_kafka')
def inside(name, roots):
    return any(name == r or name.startswith(r + '.') for r in roots)
for source, target in edges:
    if inside(source, common + ('agentweave.ports',)):
        assert not inside(target, native + ('agentweave.adapters', 'agentweave._runtime', 'agentweave.worker', 'agentweave._infrastructure')), (source,target)
    if inside(source, ('agentweave.adapters',)):
        assert not inside(target, ('agentweave._runtime', 'agentweave.worker', 'agentweave._infrastructure')), (source,target)
    if inside(source, ('agentweave.client',)):
        assert not inside(target, ('agentweave._runtime', 'agentweave.worker')), (source,target)
assert ('agentweave.models', 'agentweave.messages') in edges
assert ('agentweave._records', 'agentweave.messages') in edges
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=os.environ | {'PYTHONPATH':str(Path(__file__).resolve().parents[2] / 'src')})
    assert result.returncode == 0, result.stderr
