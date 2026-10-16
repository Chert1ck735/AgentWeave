"""Build the local wheelhouse: our two wheels and every dependency wheel pinned by the hash lock.

    python scripts/build_wheelhouse.py

Writes dist/wheelhouse/ (agentweave, agentweave-dev), dist/wheelhouse/deps/ (binary
wheels verified against requirements/py312.lock) and dist/wheelhouse/SHA256SUMS.
Dependencies are fetched once, here; installing from the wheelhouse needs no network:

    uv pip install --offline --no-index --find-links dist/wheelhouse --find-links dist/wheelhouse/deps \\
        -c requirements/py312.lock "agentweave[dev]"

Nothing is uploaded anywhere: publishing to an index needs a separate, explicit decision.
"""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / 'requirements' / 'py312.lock'
PIP = 'pip==25.2'


def run(*args):
    subprocess.run(args, cwd=REPO, check=True)


def build(target: Path = REPO / 'dist' / 'wheelhouse') -> Path:
    uv = shutil.which('uv') or 'uv'
    deps = target / 'deps'
    deps.mkdir(parents=True, exist_ok=True)
    for old in target.glob('agentweave*.whl'):
        old.unlink()
    for source in (REPO, REPO / 'packages' / 'agentweave-dev'):
        run(uv, 'build', '--wheel', '--out-dir', str(target), str(source))
    # Binary wheels only, each checked against the lock's hashes; no dependency resolution here.
    run(uv, 'run', '--no-project', '--with', PIP, '--python', '3.12', 'python', '-m', 'pip', 'download',
        '--require-hashes', '--only-binary=:all:', '--no-deps', '-r', str(LOCK), '-d', str(deps))
    lines = [f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(target)}'
             for path in sorted(target.rglob('*.whl'))]
    (target / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    return target


if __name__ == '__main__':
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'dist' / 'wheelhouse'))
