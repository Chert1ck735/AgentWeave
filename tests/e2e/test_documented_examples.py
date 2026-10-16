"""The documentation shows only commands the CLI has, and every example validates and runs offline."""
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from agentweave.cli.app import build_parser

REPO = Path(__file__).resolve().parents[2]
DOCS = ['README.md', 'docs/quickstart.md', 'docs/configuration.md', 'docs/tools.md', 'docs/coordination.md',
        'docs/recovery.md']
ENV = os.environ | {'PYTHONPATH': os.pathsep.join([str(REPO / 'src'), str(REPO / 'packages/agentweave-dev/src'),
                                                    str(REPO)])}
FENCE = re.compile(r'^```([a-z]*)\n(.*?)^```', re.S | re.M)  # every fenced block, paired in order
SHELL = ('', 'bash', 'sh', 'console')
WORD = re.compile(r'^[a-z][a-z0-9_.:-]*$')
PLACEHOLDER = re.compile(r'<([a-z-]+)>')


def agentweave(*args, root=REPO, env=ENV):
    return subprocess.run([sys.executable, '-m', 'agentweave', '--root', str(root), '--json', *args],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=60)


def documented_commands():
    commands = []
    for name in DOCS:
        for language, block in FENCE.findall((REPO / name).read_text(encoding='utf-8')):
            if language not in SHELL:
                continue
            for line in block.splitlines():
                line = line.strip()
                if line.startswith('agentweave '):
                    commands.append((name, line))
    return commands


def test_every_document_exists_and_shows_cli_commands():
    missing = [name for name in DOCS if not (REPO / name).is_file()]
    assert not missing, missing
    assert len(documented_commands()) >= 15


def test_every_documented_command_exists():
    unknown = []
    for name, line in documented_commands():
        words = []
        for word in line.split()[1:]:
            if word.startswith('-') or not WORD.match(word):
                break
            words.append(word)
        if agentweave(*words[:3], '--help').returncode != 0:
            unknown.append((name, line))
    assert not unknown, unknown


def test_every_documented_command_parses_with_its_options():
    rejected = []
    for name, line in documented_commands():
        argv = shlex.split(PLACEHOLDER.sub(lambda m: '1' if m[1] == 'n' else m[1], line))[1:]
        try:
            build_parser(json_mode=False).parse_args(argv)
        except SystemExit:
            rejected.append((name, line))
    assert not rejected, rejected


@pytest.mark.parametrize('target', ['examples.counter:counter', 'examples.calculator:add', 'examples.assistant:assistant'])
def test_every_python_example_validates_without_infrastructure(target):
    result = agentweave('agent', 'validate', target, env=ENV | {'OPENAI_API_KEY': 'not-a-real-key'})
    assert result.returncode == 0, (result.stdout, result.stderr)


@pytest.mark.parametrize('example, names', [('http_tool.toml', ['weather', 'forecaster']),
                                            ('mcp_toolbox.toml', ['support'])])
def test_every_configuration_example_validates(tmp_path, example, names):
    shutil.copy(REPO / 'examples' / example, tmp_path / 'agentweave.toml')
    for name in names:
        result = agentweave('agent', 'validate', name, root=tmp_path, env=ENV | {'OPENAI_API_KEY': 'not-a-real-key'})
        assert result.returncode == 0, (name, result.stdout, result.stderr)
    assert not (tmp_path / '.agentweave' / 'dev').exists()  # validation starts nothing


def test_the_assistant_example_runs_offline_without_a_key():
    env = {key: value for key, value in ENV.items() if not key.endswith('_API_KEY')}
    result = subprocess.run([sys.executable, '-m', 'examples.assistant', '--offline'], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip()
