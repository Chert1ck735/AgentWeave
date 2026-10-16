"""Packaged native broker with explicit, disposable, per-epoch SQLite storage."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from uuid import UUID,uuid4
from .processes import (InfrastructureError,ProcessRegistry,atomic_json,capture,
                        owner_environment,private_dir,stop_unrecorded_child)


def packaged_binary():
    try:
        from agentweave_dev import resolve_broker_bin
    except ImportError as exc:
        raise InfrastructureError('dev_extra_missing', 'install the local binary extra: pip install "agentweave[dev]"') from exc
    try:
        binary = resolve_broker_bin().resolve()
        result = subprocess.run([str(binary),'--version'],capture_output=True,text=True,timeout=10,
                                env={'PATH':os.defpath})
        if result.returncode or '0.6.0' not in result.stdout:
            raise RuntimeError('packaged broker version mismatch')
        return binary,result.stdout.strip()
    except (RuntimeError,OSError,subprocess.SubprocessError) as exc:
        raise InfrastructureError('dev_binary_unavailable',str(exc)) from exc


def ready(url,timeout=1):
    from confluent_kafka.admin import AdminClient
    import confluent_kafka as ck
    try:
        metadata = AdminClient({'bootstrap.servers':url,'socket.timeout.ms':1000,'log_level':0}).list_topics(timeout=timeout)
        return bool(metadata.brokers)
    except ck.KafkaException:
        return False


def cleanup_epoch(record,paths):
    """Delete only the recorded private epoch, after the recorded process is gone."""
    registry = ProcessRegistry(paths.root,paths.profile)
    if registry.running(record):
        raise InfrastructureError('ownership_mismatch','cannot remove live broker storage')
    try:
        epoch = str(UUID(record.broker_epoch))
        instance = str(UUID(record.instance_id))
    except ValueError as exc:
        raise InfrastructureError('ownership_mismatch','epoch identity is not a canonical UUID') from exc
    path = Path(record.epoch_dir)
    canonical_epochs = paths.root.resolve()/'.agentweave'/paths.profile/'epochs'
    if (paths.profile not in ('dev','server') or
        epoch != record.broker_epoch or instance != record.instance_id or
        paths.epochs.resolve() != canonical_epochs or path.resolve() != canonical_epochs/epoch or
        path != paths.epochs/epoch or
        any(p.is_symlink() for p in (paths.base,paths.profile_dir,paths.epochs,path))):
        raise InfrastructureError('ownership_mismatch','epoch directory proof failed')
    if not path.exists(): return
    try:
        marker = json.loads((path/'owner.json').read_text())
    except (OSError,ValueError) as exc:
        raise InfrastructureError('ownership_mismatch','epoch owner marker is missing or invalid') from exc
    expected = {'root':str(paths.root),'project_id':record.project_id,'instance_id':record.instance_id,
                'broker_epoch':record.broker_epoch}
    if marker != expected or path.stat().st_uid != os.getuid():
        raise InfrastructureError('ownership_mismatch','epoch owner proof failed')
    shutil.rmtree(path)


def reconcile_dead(record,registry,paths):
    """Forget a broker whose process is gone; never signals the PID's new occupant."""
    registry.require_dead(record)
    if record.epoch_dir and Path(record.epoch_dir).exists() and registry.directory_in_use(record.epoch_dir):
        raise InfrastructureError('ownership_mismatch','epoch storage is still used by a live process')
    # A dead owner may only authorize removal of its specifically marked epoch.
    cleanup_epoch(record,paths)
    registry.remove(record)


def _reuse_running(old,config,registry,paths,binary,version):
    if not registry.verify(old) or old.version != version or str(binary) != old.argv[0]:
        raise InfrastructureError('ownership_mismatch','existing broker owner/version proof failed; '
                                  'run "agentweave infra down" before upgrading')
    adopt = config.broker_epoch is None and config.broker_url is None
    if not adopt and (old.broker_epoch != config.broker_epoch or old.broker_url != config.broker_url):
        raise InfrastructureError('ownership_mismatch','broker endpoint/epoch differs from configuration')
    if not ready(old.broker_url,3):
        raise InfrastructureError('broker_unavailable','owned broker is not ready')
    if adopt:
        # An interrupted first run saved the verified broker before its config.
        return config.model_copy(update={'broker_url':old.broker_url,'broker_epoch':old.broker_epoch,
            'database_url':'sqlite:///'+str(paths.journal)})
    return config


def _discard_failed_start(child,record,registry,paths,epoch_dir):
    """Undo a broker start; the Popen handle proves ownership even without a record."""
    if child is not None:
        stop_unrecorded_child(child)
    if record is not None:
        cleanup_epoch(record,paths)
        registry.remove(record)
    elif epoch_dir.exists():
        shutil.rmtree(epoch_dir)


def ensure_broker(config,paths):
    binary,version = packaged_binary()
    registry = ProcessRegistry(config.root,config.profile)
    brokers = [x for x in registry.records() if x.logical_id == '$broker']
    for old in brokers:
        if registry.running(old):
            return _reuse_running(old,config,registry,paths,binary,version)
        reconcile_dead(old,registry,paths)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        url = '127.0.0.1:'+str(sock.getsockname()[1])
    epoch,instance = str(uuid4()),str(uuid4())
    epoch_dir = private_dir(paths.epochs/epoch)
    record = None
    child = None
    try:
        atomic_json(epoch_dir/'owner.json',{'root':str(paths.root),'project_id':config.project_id,
            'instance_id':instance,'broker_epoch':epoch})
        argv = [str(binary),'--silent','--cluster-id',config.network_id,'--listener-url','tcp://'+url,
                '--advertised-listener-url','tcp://'+url,'--storage-engine','sqlite:///broker.sqlite']
        env = {key:value for key,value in os.environ.items() if key in ('PATH','LANG','LC_ALL')}
        env.update(owner_environment(config,instance)); env['RUST_LOG']='warn'
        with (paths.logs/'broker.log').open('ab') as log:
            os.chmod(log.name,0o600)
            child = subprocess.Popen(argv,cwd=epoch_dir,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True,umask=0o077)
        record = capture(child,config,'$broker',instance,argv,epoch_dir,
            broker_epoch=epoch,epoch_dir=str(epoch_dir),broker_url=url,version=version)
        registry.save(record)
        deadline = time.monotonic()+20
        while time.monotonic()<deadline:
            if child.poll() is not None:
                raise InfrastructureError('broker_start_failed','broker exited; see private broker log')
            if ready(url):
                return config.model_copy(update={'broker_url':url,'broker_epoch':epoch,
                    'database_url':'sqlite:///'+str(paths.journal)})
            time.sleep(.1)
        raise InfrastructureError('broker_start_timeout','broker readiness deadline exhausted')
    except BaseException as error:
        try:
            _discard_failed_start(child,record,registry,paths,epoch_dir)
        except Exception as cleanup_error:
            # Report why the start failed; the cleanup failure is secondary.
            error.add_note(f'broker start cleanup also failed: {cleanup_error}')
        raise

