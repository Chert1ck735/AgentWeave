"""Resolve only the integrity-checked executable owned by this package."""
import atexit
import ctypes
from contextlib import ExitStack
import hashlib
from importlib import resources
import json
from pathlib import Path
import platform

_extractions = ExitStack()
atexit.register(_extractions.close)
_extracted_path = None


def resolve_broker_bin() -> Path:
    """Return the packaged Linux x86_64 broker, retaining extracted resources."""
    global _extracted_path
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('agentweave-dev requires Linux x86_64; install its supported wheel')
    libc, libc_version = platform.libc_ver()
    try:
        glibc = tuple(map(int, libc_version.split('.')))
    except ValueError:
        glibc = ()
    if libc != 'glibc' or glibc < (2, 39):
        raise RuntimeError('agentweave-dev requires glibc >= 2.39; musl and older glibc are unsupported')
    for library in ('libgcc_s.so.1', 'libm.so.6', 'libc.so.6'):
        try:
            ctypes.CDLL(library)
        except OSError as exc:
            raise RuntimeError(f'Required broker native library {library} is unavailable: {exc}; install it using your OS administrator') from exc
    if not Path('/lib64/ld-linux-x86-64.so.2').is_file():
        raise RuntimeError('Required broker loader /lib64/ld-linux-x86-64.so.2 is unavailable')
    package = resources.files('agentweave_dev')
    try:
        manifest = json.loads(package.joinpath('manifest.json').read_text())
        resource = package.joinpath('_bin', 'tansu')
        if not resource.is_file():
            raise RuntimeError('Packaged broker is missing; install the agentweave-dev binary wheel')
        if hashlib.sha256(resource.read_bytes()).hexdigest() != manifest['sha256']:
            raise RuntimeError('Packaged broker SHA256 integrity verification failed')
        if isinstance(resource, Path):
            path = resource
        else:
            if _extracted_path is None:
                _extracted_path = _extractions.enter_context(resources.as_file(resource))
                _extracted_path.chmod(_extracted_path.stat().st_mode | 0o700)
            path = _extracted_path
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
            raise RuntimeError('Extracted broker SHA256 integrity verification failed')
        if not path.stat().st_mode & 0o111:
            raise RuntimeError('Packaged broker is not executable; reinstall agentweave-dev')
        return path
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise RuntimeError('Packaged broker manifest is missing or invalid') from exc
