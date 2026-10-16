"""Package the pinned official release without compiling or altering its ELF.

Download the archive during build preparation from the manifest URL, then pass
its local path. Runtime package resolution never fetches executable code.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

VERSION = '0.6.0'
SOURCE_SHA = '079c4c334a0e9240933d208ecd38c1127526e89c'
ARCHIVE_SHA256 = 'e3f5ddb6be4f92fb082cae51c58a7745c7ec3e16a87bc4c984e4ddb5a552e1be'
BINARY_SHA256 = '92f6c24778f97b69d2f520b4cc7e9563fc1af81768fcd8d9d9a1fcaac7551d75'
READELF_ENV = os.environ | {'LC_ALL': 'C'}  # readelf translates its labels; the checks below parse English
RELEASE_URL = 'https://github.com/nisshi-io/nisshi/releases/download/v0.6.0/tansu-x86_64-unknown-linux-gnu.tar.gz'
ROOT = Path(__file__).resolve().parents[1]


def readelf(option: str, binary: Path) -> str:
    return subprocess.check_output(['readelf', option, str(binary)], text=True, env=READELF_ENV)


def package_release(archive: Path, package_dir: Path):
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError('Release archive SHA256 differs from pinned GitHub asset digest')
    required = {'bin/tansu', 'LICENSE', 'THIRD-PARTY-LICENSE.html'}
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix('./')
            if name in required:
                if not member.isfile() or name in files:
                    raise RuntimeError(f'Invalid release member: {name}')
                files[name] = tar.extractfile(member).read()
    if set(files) != required:
        raise RuntimeError('Release is missing binary or required license notices')
    if hashlib.sha256(files['bin/tansu']).hexdigest() != BINARY_SHA256:
        raise RuntimeError('Release executable SHA256 differs from pinned binary')
    with tempfile.TemporaryDirectory(prefix='aw-elf-check-') as temporary:
        binary = Path(temporary) / 'tansu'
        binary.write_bytes(files['bin/tansu'])
        header = readelf('-h', binary)
        dynamic = readelf('-d', binary)
        symbols = readelf('--version-info', binary)
        interpreter = readelf('-l', binary)
    if 'Advanced Micro Devices X86-64' not in header:
        raise RuntimeError('Release ELF architecture differs from Linux x86_64')
    dependencies = re.findall(r'Shared library: \[(.*?)\]', dynamic)
    if set(dependencies) != {'libgcc_s.so.1', 'libm.so.6', 'libc.so.6', 'ld-linux-x86-64.so.2'}:
        raise RuntimeError(f'Unexpected ELF dependencies: {dependencies}')
    required_glibc = max(re.findall(r'GLIBC_([\d.]+)', symbols), key=lambda s: tuple(map(int, s.split('.'))))
    if required_glibc != '2.39' or '/lib64/ld-linux-x86-64.so.2' not in interpreter:
        raise RuntimeError('Release loader or glibc requirement changed')
    package = package_dir / 'src/agentweave_dev'
    (package / '_bin').mkdir(parents=True, exist_ok=True)
    binary = package / '_bin/tansu'
    binary.write_bytes(files['bin/tansu'])
    binary.chmod(0o755)
    licenses = package_dir / 'licenses'
    licenses.mkdir(parents=True, exist_ok=True)
    for name in ('LICENSE', 'THIRD-PARTY-LICENSE.html'):
        (licenses / name).write_bytes(files[name])
    (licenses / 'NOTICE').write_text(
        f'This package redistributes the unmodified official Tansu {VERSION} GNU release.\n'
        f'Upstream: https://github.com/nisshi-io/nisshi\nSource commit: {SOURCE_SHA}\n'
        'Apache-2.0; see LICENSE and THIRD-PARTY-LICENSE.html for third-party notices.\n')
    manifest = dict(version=VERSION, source_sha=SOURCE_SHA, target='x86_64-unknown-linux-gnu',
                    sha256=BINARY_SHA256, archive_sha256=ARCHIVE_SHA256,
                    release_url=RELEASE_URL, release_tag='v0.6.0',
                    asset_name='tansu-x86_64-unknown-linux-gnu.tar.gz', license='Apache-2.0', min_glibc=required_glibc,
                    wheel_tag='py3-none-manylinux_2_39_x86_64',
                    dynamic_dependencies=dependencies, provenance='official-prebuilt-release')
    (package / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--package-dir', type=Path, default=ROOT / 'packages/agentweave-dev')
    args = parser.parse_args()
    print(json.dumps(package_release(args.archive, args.package_dir), indent=2))
