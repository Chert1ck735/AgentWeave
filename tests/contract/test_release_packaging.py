"""Catch accidental packaging of an unverified upstream archive."""
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def test_rejects_unverified_release_archive_without_writing_output(tmp_path):
    script = Path('scripts/build_dev_broker.py').resolve()
    assert script.is_file(), 'release packaging script is missing'
    archive = tmp_path / 'forged.tar.gz'
    archive.write_bytes(b'not the official release')
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, str(script), '--archive', str(archive),
                             '--package-dir', str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'SHA256' in result.stderr
    assert not output.exists()



def test_elf_checks_read_english_labels_under_any_locale(monkeypatch):
    # readelf translates 'Shared library'; with a Russian locale no dependency was found at all.
    monkeypatch.setenv('LC_ALL', 'ru_RU.UTF-8')
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delitem(sys.modules, 'build_dev_broker', raising=False)
    import build_dev_broker
    calls = []
    monkeypatch.setattr(build_dev_broker.subprocess, 'check_output',
                        lambda argv, **options: calls.append((argv, options)) or 'Shared library: [libc.so.6]')
    assert 'Shared library' in build_dev_broker.readelf('-d', Path('/bin/sh'))
    [(argv, options)] = calls
    assert argv[0] == 'readelf' and options['env']['LC_ALL'] == 'C'
    monkeypatch.delitem(sys.modules, 'build_dev_broker')


def test_the_dev_package_ships_the_project_license_unchanged():
    root = SCRIPTS.parent
    assert (root / 'packages/agentweave-dev/licenses/AGENTWEAVE-LICENSE').read_bytes() == (root / 'LICENSE').read_bytes()
