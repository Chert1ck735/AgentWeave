"""The CLI chat keeps one conversation, shows who answers, and can reattach to it by ID."""
import json
import subprocess
import sys
from tests.e2e.test_dev_cli import CREATED, ENV, REPO, cli, ok  # noqa: F401
import pytest

APP = '''from agentweave import Agent, Peers
general = Agent('general', peers=Peers.discover())


@general.on_request
async def route(ctx, request):
    ctx.handoff('analyst', message='the user needs analysis')


analyst = Agent('analyst')


@analyst.on_request
async def answer(ctx, request):
    return {'analysed': request.input, 'context': [m.content for m in request.history]}
'''


@pytest.fixture
def chat_project(tmp_path):
    from agentweave._infrastructure.local import cleanup_epoch
    from agentweave._infrastructure.processes import ProcessRegistry
    from pathlib import Path
    root = tmp_path / 'chat'
    root.mkdir()
    (root / 'app.py').write_text(APP)
    try:
        yield root
    finally:
        for name in ('general', 'analyst'):
            cli(root, 'agent', 'stop', name, timeout=70)
        cli(root, 'infra', 'down', timeout=70)
        registry = ProcessRegistry(root)
        for record in CREATED.pop(root.resolve(), {}).values():
            if registry.running(record):
                registry.terminate(record, grace=2)
            if record.logical_id == '$broker' and Path(record.epoch_dir).exists():
                cleanup_epoch(record, registry.paths)
            registry.remove(record)


def chat(root, lines, *flags):
    result = subprocess.run([sys.executable, '-m', 'agentweave', '--root', str(root), '--json', 'agent', 'chat',
                             'general', *flags], input=''.join(json.dumps(line) + '\n' for line in lines),
                            cwd=REPO, env=ENV, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def test_chat_follows_the_handoff_and_reattaches_to_the_same_conversation(chat_project):
    ok(chat_project, 'agent', 'start', 'app:analyst')
    ok(chat_project, 'agent', 'start', 'app:general')
    opened, first, second = chat(chat_project, ['refund policy?', 'and for gifts?'])
    conversation = opened['conversation_id']
    assert opened['active_agent'] == 'general'
    assert (first['responding_agent'], first['active_agent']) == ('analyst', 'analyst')
    assert 'the user needs analysis' in first['output']['context']
    assert (second['responding_agent'], second['output']['analysed']) == ('analyst', 'and for gifts?')
    # stdin closed after the last answer: nothing was cancelled, and the chat can be reopened.
    reopened, third = chat(chat_project, ['still there?'], '--conversation', conversation)
    assert (reopened['conversation_id'], reopened['active_agent']) == (conversation, 'analyst')
    assert third['responding_agent'] == 'analyst'
