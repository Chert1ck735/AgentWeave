"""Catch unsafe packaged binary resolution and unnoticed package corruption."""
import importlib
from pathlib import Path
import pytest


def locator():
    try:
        return importlib.import_module('agentweave_dev.locator')
    except ModuleNotFoundError:
        pytest.fail('broker locator is missing')


def test_resolves_verified_executable():
    path = locator().resolve_broker_bin()
    assert path.is_file()
    assert path.stat().st_mode & 0o111


def test_rejects_unsupported_architecture(monkeypatch):
    module = locator()
    monkeypatch.setattr(module.platform, 'machine', lambda: 'aarch64')
    with pytest.raises(RuntimeError, match='x86_64'):
        module.resolve_broker_bin()


def test_rejects_modified_binary(tmp_path, monkeypatch):
    module = locator()
    package = tmp_path / 'package'
    (package / '_bin').mkdir(parents=True)
    (package / '_bin/tansu').write_bytes(b'corrupted')
    (package / 'manifest.json').write_text('{"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}')
    monkeypatch.setattr(module.resources, 'files', lambda _: package)
    with pytest.raises(RuntimeError, match='SHA256'):
        module.resolve_broker_bin()


def test_missing_binary_is_explicit(tmp_path, monkeypatch):
    module = locator()
    (tmp_path / 'manifest.json').write_text('{"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}')
    monkeypatch.setattr(module.resources, 'files', lambda _: tmp_path)
    with pytest.raises(RuntimeError, match='missing'):
        module.resolve_broker_bin()


def test_rejects_musl_before_resolving_binary(monkeypatch):
    module = locator()
    monkeypatch.setattr(module.platform, 'libc_ver', lambda: ('musl', '1.2.5'))
    with pytest.raises(RuntimeError, match='glibc'):
        module.resolve_broker_bin()


def test_rejects_older_glibc_before_resolving_binary(monkeypatch):
    module = locator()
    monkeypatch.setattr(module.platform, 'libc_ver', lambda: ('glibc', '2.38'))
    with pytest.raises(RuntimeError, match='glibc'):
        module.resolve_broker_bin()


def test_rejects_non_linux(monkeypatch):
    module = locator()
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    with pytest.raises(RuntimeError, match='Linux'):
        module.resolve_broker_bin()


def test_missing_native_library_has_actionable_diagnostic(monkeypatch):
    import ctypes
    module = locator()
    def unavailable(name):
        raise OSError('library not found')
    monkeypatch.setattr(ctypes, 'CDLL', unavailable)
    with pytest.raises(RuntimeError, match='libgcc_s.so.1'):
        module.resolve_broker_bin()


def test_archive_resource_is_extracted_and_retained_without_changing_archive(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile
    module = locator()
    archive = tmp_path / 'package.zip'
    content = b'package-owned executable fixture'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('agentweave_dev/_bin/tansu', content)
        output.writestr('agentweave_dev/manifest.json', json.dumps({'sha256': hashlib.sha256(content).hexdigest()}))
    original_archive = archive.read_bytes()
    with zipfile.ZipFile(archive) as source:
        resource = zipfile.Path(source, 'agentweave_dev/')
        monkeypatch.setattr(module.resources, 'files', lambda _: resource)
        path = module.resolve_broker_bin()
        assert path.read_bytes() == content
        assert path.stat().st_mode & 0o100
        assert module.resolve_broker_bin() == path
    assert path.exists(), 'resource lifetime must exceed the resource reader context'
    assert archive.read_bytes() == original_archive
