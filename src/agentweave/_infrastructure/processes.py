"""Linux process ownership records; PID alone is never authority to signal."""
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
import psutil


class InfrastructureError(RuntimeError):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code = code
        self.details = details  # structural diagnostics only: never input values or secrets


def private_dir(path):
    path = Path(path)
    if path.is_symlink():
        raise InfrastructureError('ownership_mismatch', 'private data directory must not be a symlink')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid():
        raise InfrastructureError('ownership_mismatch', 'data directory belongs to another owner')
    path.chmod(0o700)
    return path


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def executable_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass(frozen=True)
class ProcessRecord:
    project_id: str
    profile: str
    logical_id: str
    instance_id: str
    pid: int
    create_time: float
    executable_hash: str
    root: str = ''
    cwd: str = ''
    argv: list[str] = field(default_factory=list)
    target: str = ''
    network_id: str = ''
    broker_epoch: str = ''
    epoch_dir: str = ''
    broker_url: str = ''
    version: str = ''
    overrides: dict = field(default_factory=dict)


class ProcessRegistry:
    def __init__(self, root, profile='dev'):
        self.root = Path(root).resolve()
        if profile not in ('dev','server'):
            raise InfrastructureError('profile_invalid','profile must be dev or server')
        self.profile = profile
        from .bootstrap import Paths
        self.paths = Paths(self.root,profile)

    def save(self, record):
        if record.profile != self.profile or record.root not in ('', str(self.root)):
            raise InfrastructureError('ownership_mismatch','record belongs to another project')
        private_dir(self.paths.registry)
        # UUID instance IDs are also the sole registry filenames.
        from uuid import UUID
        name = str(UUID(record.instance_id))
        atomic_json(self.paths.registry/(name+'.json'), asdict(record))

    def records(self):
        if not self.paths.registry.exists():
            return []
        records = []
        for path in sorted(self.paths.registry.glob('*.json')):
            try:
                records.append(ProcessRecord(**json.loads(path.read_text())))
            except (ValueError,TypeError,OSError) as exc:
                raise InfrastructureError('registry_invalid','invalid process registry') from exc
        return records

    def verify(self, record):
        try:
            from ..config import resolve
            config = resolve(self.root,self.profile)
            if (record.root != str(self.root) or record.profile != self.profile or
                    record.project_id != config.project_id or record.network_id != config.network_id):
                return False
            process = psutil.Process(record.pid)
            if (process.status() == psutil.STATUS_ZOMBIE or process.uids().real != os.getuid() or
                    process.create_time() != record.create_time or process.cwd() != record.cwd or
                    process.cmdline() != record.argv or executable_hash(process.exe()) != record.executable_hash):
                return False
            environment = process.environ()
            return all(environment.get(key) == value for key,value in {
                'AGENTWEAVE_PROJECT_ID':record.project_id,'AGENTWEAVE_PROFILE':record.profile,
                'AGENTWEAVE_ROOT':str(self.root),'AGENTWEAVE_INSTANCE_ID':record.instance_id}.items())
        except (psutil.Error,OSError,ValueError):
            return False

    def remove(self, record):
        from uuid import UUID
        (self.paths.registry/(str(UUID(record.instance_id))+'.json')).unlink(missing_ok=True)
        (self.paths.registry/(str(UUID(record.instance_id))+'.ready')).unlink(missing_ok=True)
        (self.paths.registry/(str(UUID(record.instance_id))+'.exit')).unlink(missing_ok=True)

    @staticmethod
    def pid_alive(pid):
        try: return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess: return False

    def require_dead(self, record):
        """Confirm a non-running record's process is gone, without signalling anyone.

        A PID reused by an unrelated process proves the original exited. A
        process carrying the record's own owner identity means the record
        itself is wrong, so its storage must not be treated as abandoned.
        """
        if self.running(record):
            raise InfrastructureError('ownership_mismatch','recorded process is still running')
        try:
            process = psutil.Process(record.pid)
            if process.status() == psutil.STATUS_ZOMBIE or process.uids().real != os.getuid():
                return
            environment = process.environ()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        if (environment.get('AGENTWEAVE_INSTANCE_ID') == record.instance_id and
                environment.get('AGENTWEAVE_PROJECT_ID') == record.project_id):
            raise InfrastructureError('ownership_mismatch','record does not match its own live process')

    @staticmethod
    def directory_in_use(path):
        """True when any process of this user works inside path, such as a live broker."""
        target = str(Path(path).resolve())
        for process in psutil.process_iter(['uids', 'cwd']):
            info = process.info
            if info['uids'] and info['uids'].real == os.getuid() and info['cwd'] == target:
                return True
        return False

    def running(self, record):
        try:
            process = psutil.Process(record.pid)
            return process.create_time() == record.create_time and process.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def terminate(self, record, *, drain=False, grace=30):
        if not self.verify(record):
            if not self.pid_alive(record.pid): return False
            raise InfrastructureError('ownership_mismatch','process owner proof failed; no signal sent')
        escalated = False
        for sig, duration in [(signal.SIGUSR1 if drain else signal.SIGINT,grace),
                              (signal.SIGTERM,2),(signal.SIGKILL,2)]:
            if not self.running(record):
                return escalated
            if sig in (signal.SIGTERM,signal.SIGKILL): escalated = True
            if not self.verify(record):
                if not self.pid_alive(record.pid): return escalated
                raise InfrastructureError('ownership_mismatch','process owner changed; no signal sent')
            # pidfd pins the process identity across the final verify/signal race.
            try:
                fd = os.pidfd_open(record.pid)
            except ProcessLookupError:
                return
            try:
                if not self.verify(record):
                    if not self.pid_alive(record.pid): return escalated
                    raise InfrastructureError('ownership_mismatch','process owner changed; no signal sent')
                signal.pidfd_send_signal(fd,sig)
            finally:
                os.close(fd)
            deadline = time.monotonic()+duration
            while self.running(record) and time.monotonic()<deadline:
                time.sleep(.05)
        if self.running(record):
            raise InfrastructureError('stop_timeout','owned process has not exited')
        return escalated


@contextmanager
def file_lock(path):
    if path.is_symlink():
        raise InfrastructureError('ownership_mismatch','lock must not be a symlink')
    fd = os.open(path,os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def owner_environment(config,instance_id):
    return {'AGENTWEAVE_PROJECT_ID':config.project_id,'AGENTWEAVE_PROFILE':config.profile,
            'AGENTWEAVE_ROOT':str(config.root),'AGENTWEAVE_INSTANCE_ID':instance_id}


def capture(process, config, logical_id, instance_id, argv, cwd, **extra):
    child = psutil.Process(process.pid)
    return ProcessRecord(config.project_id,config.profile,logical_id,instance_id,process.pid,
        child.create_time(),executable_hash(child.exe()),root=str(config.root),cwd=str(cwd),
        argv=argv,network_id=config.network_id,**extra)


def stop_unrecorded_child(child):
    """Stop a child this process spawned before its ownership record was saved.

    The live Popen handle is the proof of ownership, so no PID lookup is needed.
    """
    for stop, wait in ((child.terminate, 2), (child.kill, 5)):
        if child.poll() is not None:
            return
        stop()
        try:
            child.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
