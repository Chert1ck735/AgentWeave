"""Worker subprocess launch, readiness and independent bounded supervision."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4
from .._infrastructure.bootstrap import Bootstrap, Paths
from .._infrastructure.processes import (InfrastructureError, ProcessRegistry, atomic_json,
    capture,file_lock,owner_environment,stop_unrecorded_child)


def _preflight_model(spec):
    """Check the key reference and adapter availability before any infrastructure I/O.

    Purely structural: it builds no client, so it also runs inside the async worker.
    """
    from ..errors import ConfigError,DefinitionError
    from ..providers import check_available
    key = spec.api_key_env
    if key and not os.environ.get(key):
        raise InfrastructureError('model_key_missing','required model key environment variable is absent: '+key)
    try:
        check_available(spec,os.environ)
    except (ConfigError,DefinitionError) as exc:
        raise InfrastructureError('model_unavailable',str(exc)) from exc


TOOL_OVERRIDES = frozenset(('namespace','concurrency'))


def safe_details(exc):
    """Where a definition is wrong, without echoing its values: they may hold secrets.

    Field errors come from the validation error in the cause chain; an SDK
    configuration error with no cause is our own message. Anything else,
    such as an exception raised by user module code, gives no details.
    """
    from pydantic import ValidationError
    from ..errors import ConfigError,DefinitionError
    seen = exc
    while seen is not None:
        if isinstance(seen,ValidationError):
            return [{'loc':[str(part) for part in item['loc']],'msg':item['msg']}
                    for item in seen.errors(include_input=False,include_url=False)]
        seen = seen.__cause__
    if isinstance(exc,(ConfigError,DefinitionError)) and exc.__cause__ is None:
        return [{'loc':[],'msg':str(exc)}]
    return None


def load_definition(target,root,overrides=None):
    """Load an Agent, or a ToolDefinition/ToolboxDefinition served as a tool service."""
    from ..definitions import McpToolboxSpec, ToolDefinition, ToolboxDefinition
    from ..definitions.agent import Agent
    from ..errors import DefinitionError
    try:
        from ..definitions import load_definition as resolve_definition
        definition = resolve_definition(target,root)
        if isinstance(definition,(ToolDefinition,ToolboxDefinition,McpToolboxSpec)):
            if set(overrides or {}) - TOOL_OVERRIDES:
                raise DefinitionError('tool services accept only --namespace and --concurrency')
            from .._runtime.tool_service import published_tools
            published_tools(definition,(overrides or {}).get('namespace'),root)
            return definition
        if not isinstance(definition,Agent): raise TypeError('target is not an Agent')
        if 'namespace' in (overrides or {}):
            raise DefinitionError('--namespace applies to tool services')
        if overrides:
            from ..config import _resolve_agent
            definition = _resolve_agent(definition,root,overrides)
        from ..config import check_model_capabilities
        check_model_capabilities(definition)
        if definition.model is not None:
            _preflight_model(definition.model)
        elif not definition.handlers and not definition.request_handler:
            raise DefinitionError('Agent needs a model, a request or an event handler')
        return definition
    except InfrastructureError:
        raise
    except Exception as exc:
        from ..errors import CapabilityMismatch
        if isinstance(exc,CapabilityMismatch):
            raise InfrastructureError('capability_mismatch',str(exc)) from exc
        # Module exceptions may contain secrets. Keep diagnostics structural.
        raise InfrastructureError('definition_invalid','cannot load/validate Agent or tool target; expected module:object',
                                  details=safe_details(exc)) from exc


def is_tool_record(record):
    """Tool service records always carry their resolved namespace."""
    return 'namespace' in record.overrides


def _service(definition,root,overrides):
    """The registry name of a launch and its overrides, with a tool's namespace resolved."""
    from ..definitions.agent import Agent
    if isinstance(definition,Agent):
        return definition.name,overrides
    from ..definitions import McpToolboxSpec, ToolDefinition
    from .._runtime.tool_service import published_tools
    if isinstance(definition,McpToolboxSpec):  # its tools are known only once the server runs
        namespace = overrides.get('namespace') or definition.namespace
        return namespace,overrides|{'namespace':namespace}
    tools = published_tools(definition,overrides.get('namespace'),root)
    first = next(iter(tools))
    namespace = first.rsplit('.',1)[0]
    # A single tool is registered under its tool ID, a toolbox under its namespace.
    return (first if isinstance(definition,ToolDefinition) else namespace),overrides|{'namespace':namespace}


def start_worker(target,config,detach=False,*,replica=False,overrides=None):
    overrides = dict(overrides or {})
    definition = load_definition(target,config.root,overrides)
    name,overrides = _service(definition,config.root,overrides)
    registry = ProcessRegistry(config.root,config.profile)
    paths = Paths(config.root,config.profile)
    with file_lock(paths.profile_dir/'workers.lock'):
        for old in registry.records():
            if old.logical_id != name: continue
            if registry.running(old):
                if not registry.verify(old):
                    raise InfrastructureError('ownership_mismatch','worker owner proof failed')
                if not replica:
                    raise InfrastructureError('already_running','already running; use --replica for an additional instance')
            else:
                registry.remove(old)
        instance = str(uuid4())
        argv = [sys.executable,'-m','agentweave','--root',str(config.root),'--profile',config.profile,
                '_worker',target,instance,'--overrides',json.dumps(overrides,separators=(',',':'))]
        env = os.environ.copy()
        env.update(owner_environment(config,instance))
        # sys.path may have been supplied by an SDK host rather than PYTHONPATH.
        env['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(Path(__file__).resolve().parents[2]),
            *filter(None,env.get('PYTHONPATH','').split(os.pathsep))]))
        with (paths.logs/(instance+'.log')).open('ab') as log:
            os.chmod(log.name,0o600)
            child = subprocess.Popen(argv,cwd=config.root,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True)
        record = None
        try:
            record = capture(child,config,name,instance,argv,config.root,target=target,
                             broker_epoch=config.broker_epoch or '',overrides=overrides)
            registry.save(record)
            deadline = time.monotonic()+30
            while time.monotonic()<deadline:
                if child.poll() is not None:
                    raise InfrastructureError('worker_start_failed','worker exited before readiness; see private worker log')
                ready = paths.registry/(instance+'.ready')
                if ready.exists():
                    value = json.loads(ready.read_text())
                    if (registry.verify(record) and value.get('broker_epoch') == config.broker_epoch and
                        value.get('instance_id') == instance and 0 <= time.time()-value.get('heartbeat_at',0)<15):
                        return record
                time.sleep(.05)
            raise InfrastructureError('worker_start_timeout','fresh worker heartbeat deadline exhausted')
        except BaseException as error:
            try:
                if record is not None and registry.running(record):
                    try: registry.terminate(record)  # graceful stop first
                    except InfrastructureError: pass  # the handle below still stops it
                # The Popen handle proves ownership whether or not a record was saved.
                stop_unrecorded_child(child)
                if record is not None: registry.remove(record)
            except Exception as cleanup_error:
                error.add_note(f'worker start cleanup also failed: {cleanup_error}')
            raise


async def worker_main(target,instance,root,profile,overrides=None):
    from ..config import resolve
    # Validate trusted project code before opening native adapter threads.
    resources = None
    outcome = {'instance_id':instance,'error':None,'operation':'stop'}
    try:
        config = resolve(root,profile)
        definition = load_definition(target,root,overrides)
        resources = await Bootstrap.open_resources(config)
        runner,names = _runner(definition,instance,resources,overrides or {})
        await _serve_worker(runner,names,instance,resources,outcome)
    except BaseException as exc:
        outcome['error'] = type(exc).__name__
        raise
    finally:
        try:
            if resources is not None: await resources.close()
        except BaseException as exc:
            outcome['error'] = type(exc).__name__
            raise
        finally:
            atomic_json(Paths(root,profile).registry/(instance+'.exit'),outcome)


def _runner(definition,instance,resources,overrides):
    """A Worker for an Agent or a ToolWorker for tools, and the names it announces."""
    from ..definitions.agent import Agent
    if isinstance(definition,Agent):
        from ..worker import Worker
        worker = Worker(definition,resources.config,resources.store,resources.bus,resources.directory)
        worker.instance_id = instance
        worker.pump.instance_id = instance
        return worker,(definition.name,)
    from .._runtime.tool_service import ToolWorker
    worker = ToolWorker(definition,resources.config,resources.store,resources.bus,resources.directory,
        namespace=overrides.get('namespace'),concurrency=overrides.get('concurrency',1),instance_id=instance)
    return worker,worker.logical_ids


async def _serve_worker(worker,names,instance,resources,outcome):
    config = resources.config
    loop = asyncio.get_running_loop()
    requested = asyncio.Event()
    drain = False
    def stop(mode=False):
        nonlocal drain
        drain = mode
        outcome['operation'] = 'drain' if mode else 'stop'
        requested.set()
    loop.add_signal_handler(signal.SIGINT,stop)
    loop.add_signal_handler(signal.SIGTERM,stop)
    loop.add_signal_handler(signal.SIGUSR1,stop,True)
    run = asyncio.create_task(worker.run())
    stopping = asyncio.create_task(requested.wait())
    ready_wait = asyncio.create_task(worker.ready.wait())
    try:
        done,_ = await asyncio.wait([run,ready_wait,stopping],return_when=asyncio.FIRST_COMPLETED)
        if run in done: await run
        if stopping in done: return
        await ready_wait
        async with asyncio.timeout(20):
            while not all(any(x.instance_id==instance for x in resources.directory.snapshot().get(name,()))
                          for name in names):
                if run.done(): await run
                await asyncio.sleep(.02)
        atomic_json(Paths(config.root,config.profile).registry/(instance+'.ready'),{
            'instance_id':instance,'broker_epoch':config.broker_epoch,'heartbeat_at':time.time()})
        done,_ = await asyncio.wait([run,stopping],return_when=asyncio.FIRST_COMPLETED)
        if stopping in done:
            if drain: await worker.drain()
            else: await worker.stop()
        await run
    finally:
        ready_wait.cancel(); stopping.cancel()
        await asyncio.gather(ready_wait,stopping,return_exceptions=True)
        await worker.stop()


def select_records(root,profile,name=None,instance=None):
    registry = ProcessRegistry(root,profile)
    records = [r for r in registry.records() if r.logical_id != '$broker' and
               (name is None or r.logical_id==name) and (instance is None or r.instance_id==instance)]
    if instance is not None and not records:
        raise InfrastructureError('agent_not_found','requested replica does not exist')
    return registry,records


def stop_workers(root,profile,name,instance=None,*,drain=False,only=None):
    registry,records = select_records(root,profile,name,instance)
    if only is not None:
        records = [r for r in records if only(r)]
    incomplete = False
    for record in records:
        escalated = False
        # A stale record never authorizes a signal to whoever now holds its PID.
        if registry.running(record): escalated = registry.terminate(record,drain=drain)
        else: registry.require_dead(record)
        exit_path=registry.paths.registry/(record.instance_id+'.exit')
        outcome=json.loads(exit_path.read_text()) if exit_path.exists() else None
        if drain and (escalated or outcome is None or outcome.get('error')):
            incomplete=True
        registry.remove(record)
    if incomplete:
        raise InfrastructureError('drain_incomplete','worker stopped but drain did not complete; inspect unresolved journal outcomes')
    return {'stopped':[r.instance_id for r in records], 'external_effects':'journal preserved; unknown outcomes remain unresolved'}


def supervise(record,root,profile):
    registry = ProcessRegistry(root,profile)
    interrupted = False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    previous = {sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        while registry.running(record) and not interrupted: time.sleep(.1)
        if interrupted and registry.running(record): registry.terminate(record)
        base = registry.paths.registry/record.instance_id
        # A stop requested elsewhere (agent stop) removes the record first.
        requested = interrupted or not base.with_suffix('.json').exists()
        exit_path = base.with_suffix('.exit')
        outcome = json.loads(exit_path.read_text()) if exit_path.exists() else None
        registry.remove(record)
        if not requested and (outcome is None or outcome.get('error')):
            raise InfrastructureError('worker_exited','worker exited unexpectedly; see private worker log')
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)
