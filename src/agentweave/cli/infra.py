"""Infrastructure lifecycle and diagnostics never adopt foreign processes."""
from dataclasses import asdict
from .._infrastructure.bootstrap import Bootstrap,Paths
from .._infrastructure.processes import InfrastructureError,ProcessRegistry,file_lock


def status(root,profile):
    from ..config import resolve
    registry = ProcessRegistry(root,profile)
    records = [r for r in registry.records() if r.logical_id=='$broker']
    if records:
        record = records[0]
        return asdict(record)|{'owned':registry.verify(record),'running':registry.running(record),
            'storage':'disposable per-epoch SQLite; permanent SDK journal separate'}
    try:
        config = resolve(root,profile)
        return {'owned':False,'running':False,'unmanaged':bool(config.broker_url),
                'broker_url':config.broker_url,'profile':profile}
    except Exception:
        return {'owned':False,'running':False,'unmanaged':False,'profile':profile}


def require_running(root,profile):
    """Input commands connect to running infrastructure; they never start a broker themselves."""
    paths = Paths(root,profile)
    if paths.config.exists():
        from ..config import resolve
        from ..errors import ConfigError
        try:
            resolve(root,profile)
        except ConfigError as exc:
            raise InfrastructureError('profile_invalid','the profile configuration is invalid') from exc
    value = status(root,profile)
    managed = (paths.profile_dir/'managed.json').exists()
    # A CLI-owned broker that is down stays down: only an explicitly configured broker is used as is.
    if not (value.get('running') or (value.get('unmanaged') and not managed)):
        raise InfrastructureError('infrastructure_unavailable',
            'no running infrastructure for this profile; start it with agentweave infra up or agentweave dev run')


def up(root,profile):
    config = Bootstrap(root,profile).prepare()
    return status(root,profile)|{'network_id':config.network_id}


def down(root,profile):
    paths = Paths(root,profile)
    if not paths.profile_dir.exists(): return {'stopped':[]}
    registry = ProcessRegistry(root,profile)
    with file_lock(paths.profile_dir/'bootstrap.lock'):
        brokers = [r for r in registry.records() if r.logical_id=='$broker']
        if not brokers:
            if status(root,profile).get('unmanaged'):
                raise InfrastructureError('infrastructure_unmanaged','configured broker is not CLI-owned')
            return {'stopped':[]}
        from .._infrastructure.local import cleanup_epoch,reconcile_dead
        for record in brokers:
            if not registry.running(record):
                reconcile_dead(record,registry,paths)
                continue
            registry.terminate(record,grace=2)
            cleanup_epoch(record,paths)
            registry.remove(record)
        return {'stopped':[r.instance_id for r in brokers],'journal':str(paths.journal),'journal_preserved':True}


def doctor(root,profile):
    value = status(root,profile)
    try:
        from .._infrastructure.local import packaged_binary
        binary,version = packaged_binary()
        value.update(binary_available=True,binary_version=version)
    except InfrastructureError as exc:
        value.update(binary_available=False,error_code=exc.code,message=str(exc))
    value.update(profile=profile,loopback=True,journal=str(Paths(root,profile).journal),
        broker_durability='dev events may be lost before admission; accepted tasks remain in SDK journal')
    return value
