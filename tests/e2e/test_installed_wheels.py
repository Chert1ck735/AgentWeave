"""Installed wheels, not the source checkout: a clean venv, our packages only from the local wheelhouse.

Dependencies come from the hash-locked wheelhouse (scripts/build_wheelhouse.py); agentweave and
agentweave-dev come from wheels built here; nothing comes from an index or the network. The CLI then runs with a PATH
holding only that venv: no Docker, Rust toolchain or JVM, and no network for AI calls.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
UV = shutil.which('uv') or str(Path.home() / '.local' / 'bin' / 'uv')
PYTHON = f'{sys.version_info.major}.{sys.version_info.minor}'
DEPENDENCIES = REPO / 'dist' / 'wheelhouse' / 'deps'  # scripts/build_wheelhouse.py


def uv(*args, timeout=600):
    result = subprocess.run([UV, *args], cwd=REPO, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (args, result.stdout[-2000:], result.stderr[-2000:])
    return result


class Installed:
    def __init__(self, base):
        self.base, self.wheelhouse, self.venv = base, base / 'wheelhouse', base / 'venv'
        self.project = base / 'project'

    def wheel(self, prefix):
        [path] = list(self.wheelhouse.glob(prefix + '-*.whl'))
        return path

    @property
    def bin(self):
        return self.venv / 'bin'

    def env(self):
        # Only the installed venv: no build tools, no source tree, no keys of any model provider.
        keep = {key: value for key, value in os.environ.items()
                if key in ('HOME', 'LANG', 'LC_ALL', 'TMPDIR') or key.startswith('LC_')}
        return keep | {'PATH': str(self.bin), 'PYTHONNOUSERSITE': '1'}

    def cli(self, *args, timeout=90):
        return subprocess.run([str(self.bin / 'agentweave'), '--root', str(self.project), '--json', *args],
                              cwd=self.base, env=self.env(), capture_output=True, text=True, timeout=timeout)

    def ok(self, *args, timeout=90):
        result = self.cli(*args, timeout=timeout)
        assert result.returncode == 0, (args, result.stdout, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


@pytest.fixture(scope='module')
def installed(tmp_path_factory):
    if not Path(UV).exists():
        pytest.fail('uv is required to build and install the wheels (CI installs it with setup-uv)')
    if not DEPENDENCIES.is_dir():
        pytest.fail('build the dependency wheelhouse first: python scripts/build_wheelhouse.py')
    target = Installed(tmp_path_factory.mktemp('installed'))
    # Fresh wheels of this checkout, offline: build_wheelhouse.py's online build left the build backend in uv's cache.
    for source in (REPO, REPO / 'packages' / 'agentweave-dev'):
        built = subprocess.run([UV, 'build', '--wheel', '--offline', '--out-dir', str(target.wheelhouse), str(source)],
                               cwd=REPO, capture_output=True, text=True, timeout=600)
        if built.returncode != 0:
            pytest.fail('offline wheel build failed; run python scripts/build_wheelhouse.py once (it caches the '
                        f'build backend): {built.stderr[-1500:]}')
    uv('venv', '--python', PYTHON, str(target.venv))
    python = str(target.bin / 'python')
    # No index and no network: our fresh wheels, and dependency wheels pinned by the hash lock.
    uv('pip', 'install', '--offline', '--no-index', '--python', python, '-c', 'requirements/py312.lock',
       '--find-links', str(target.wheelhouse), '--find-links', str(DEPENDENCIES), 'agentweave[dev]')
    target.project.mkdir()
    shutil.copy(REPO / 'examples' / 'counter.py', target.project / 'counter.py')
    yield target


def test_the_core_wheel_carries_no_broker_binary_or_provider_sdk(installed):
    with zipfile.ZipFile(installed.wheel('agentweave')) as wheel:
        names = wheel.namelist()
        metadata = wheel.read(next(n for n in names if n.endswith('METADATA'))).decode()
    assert not any('_bin/' in name for name in names)
    required = [line for line in metadata.splitlines() if line.startswith('Requires-Dist:')]
    unconditional = [line for line in required if 'extra ==' not in line]
    assert not any(sdk in line for line in unconditional
                   for sdk in ('openai', 'anthropic', 'google-genai', 'mcp', 'psycopg', 'aiosqlite'))
    assert installed.wheel('agentweave').name.endswith('-py3-none-any.whl')


def test_the_dev_wheel_bundles_the_pinned_broker_with_its_licenses(installed):
    wheel_path = installed.wheel('agentweave_dev')
    assert wheel_path.name.endswith('-py3-none-manylinux_2_39_x86_64.whl')
    with zipfile.ZipFile(wheel_path) as wheel:
        manifest = json.loads(wheel.read('agentweave_dev/manifest.json'))
        binary = wheel.read('agentweave_dev/_bin/tansu')
        licenses = [name for name in wheel.namelist() if '/licenses/' in name]
    assert hashlib.sha256(binary).hexdigest() == manifest['sha256']
    assert any(name.endswith('LICENSE') for name in licenses) and any(name.endswith('NOTICE') for name in licenses)



@pytest.mark.parametrize('prefix, expression', [
    ('agentweave', 'MIT'),
    ('agentweave_dev', 'MIT AND Apache-2.0')])
def test_both_wheels_carry_the_project_license(installed, prefix, expression):
    with zipfile.ZipFile(installed.wheel(prefix)) as wheel:
        names = wheel.namelist()
        metadata = wheel.read(next(n for n in names if n.endswith('METADATA'))).decode()
        texts = [wheel.read(n) for n in names if '.dist-info/licenses/' in n]
    assert f'License-Expression: {expression}' in metadata.splitlines()
    assert (REPO / 'LICENSE').read_bytes() in texts

def test_installed_dev_wheel_needs_no_build_tools(installed):
    env_path = installed.env()['PATH']
    assert all(shutil.which(tool, path=env_path) is None for tool in ('docker', 'cargo', 'rustc', 'java'))
    module = subprocess.run([str(installed.bin / 'python'), '-c', 'import agentweave, agentweave_dev; '
                             'print(agentweave.__file__)'], env=installed.env(), capture_output=True, text=True)
    assert str(installed.venv) in module.stdout  # the installed package, not the source checkout
    built = zipfile.ZipFile(installed.wheel('agentweave'))
    record = Path(module.stdout.strip()).parent / 'adapters' / 'storage' / 'sqlite_thread.py'
    assert record.read_bytes() == built.read('agentweave/adapters/storage/sqlite_thread.py')  # this build, not an index copy
    try:
        started = installed.ok('dev', 'run', 'counter:counter', '--detach', timeout=120)
        assert started[-1]['ready'] is True
        accepted, result = installed.ok('agent', 'request', 'counter', '[1]', '--session-id',
                                        '00000000-0000-4000-8000-0000000000c1')
        assert accepted['accepted'] is True and (result['status'], result['output']) == ('succeeded', 1)
        assert installed.ok('doctor')[-1]['binary_available'] is True
    finally:  # the processes this test started end with it
        installed.cli('agent', 'stop', 'counter', timeout=70)
        installed.cli('infra', 'down', timeout=70)
