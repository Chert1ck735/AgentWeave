"""Real infrastructure with explicit profile and per-test isolation."""
import os
import uuid
import pytest


def pytest_addoption(parser):
    parser.addoption('--profile', choices=('dev', 'server'), default='dev')


def pytest_configure(config):
    for marker, description in {
        'integration': 'uses real infrastructure',
        'e2e': 'end-to-end scenario',
        'live': 'requires explicitly configured external services',
    }.items():
        config.addinivalue_line('markers', f'{marker}: {description}')


@pytest.fixture
def profile(request):
    return request.config.getoption('--profile')


@pytest.fixture
def network_id():
    return 'aw-test-' + uuid.uuid4().hex


@pytest.fixture
def broker_process(profile, network_id, tmp_path):
    if profile != 'dev':
        pytest.fail('broker_process restart fixture is available only for dev profile')
    from agentweave_dev import resolve_broker_bin
    from scripts.probe_broker import BrokerProcess
    broker = BrokerProcess(resolve_broker_bin(), tmp_path, network_id)
    try:
        broker.start()
        yield broker
    finally:
        broker.stop()


@pytest.fixture
def broker_url(profile, network_id, tmp_path, request):
    if profile == 'dev':
        return request.getfixturevalue('broker_process').url
    url = os.environ.get('AGENTWEAVE_SERVER_BROKER_URL')
    if not url:
        pytest.fail('server profile requires AGENTWEAVE_SERVER_BROKER_URL')
    return url


@pytest.fixture
def candidate_report(broker_process):
    from scripts.probe_broker import check_protocol
    return check_protocol(broker_process)


class FakeClock:
    def __init__(self):
        from datetime import datetime, timezone
        self.value = datetime(2025, 1, 15, tzinfo=timezone.utc)

    def now(self):
        return self.value

    def advance(self, seconds):
        from datetime import timedelta
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def fake_clock():
    return FakeClock()


@pytest.fixture
def envelope_factory(fake_clock):
    from datetime import timedelta
    from agentweave.messages import Envelope

    def make(**changes):
        values = dict(event_id=str(uuid.uuid4()), network_id='00000000-0000-0000-0000-000000000010',
                      kind='request', sender='client.one', recipient='assistant', topic=None,
                      task_id=str(uuid.uuid4()), parent_task_id=None, correlation_id=None,
                      session_id=str(uuid.uuid4()), created_at=fake_clock.now(),
                      deadline=fake_clock.now() + timedelta(seconds=120), traceparent=None,
                      payload={'question': 'hello'})
        return Envelope(**(values | changes))
    return make


@pytest.fixture
async def store_factory(fake_clock):
    from agentweave.adapters.storage.sqlite import SQLiteStore
    stores = []

    async def make(path):
        result = SQLiteStore(path, clock=fake_clock)
        stores.append(result)
        await result.migrate()
        return result
    yield make
    for result in stores:
        await result.close()


@pytest.fixture
async def store(store_factory, tmp_path):
    return await store_factory(tmp_path / 'state.sqlite3')


@pytest.fixture
async def accepted_task(store, envelope_factory):
    event = envelope_factory()
    await store.admit(event, 'definition-v1')
    return await store.get_task(event.task_id)


@pytest.fixture
def completed_step():
    from agentweave._records import StepCommit
    return StepCommit(step_id='done', expected_version=1, status='succeeded', result={'answer': 42})


LEAK_GRACE_SECONDS = 5


def _test_processes(base):
    """Brokers and workers whose working directory or --root lies under pytest's base temp."""
    import os
    import psutil
    found = []
    for process in psutil.process_iter(['pid', 'uids', 'cmdline', 'cwd']):
        info = process.info
        if not info['uids'] or info['uids'].real != os.getuid():
            continue
        command = ' '.join(info['cmdline'] or ())
        is_broker = '_bin/tansu' in command
        is_worker = ' _worker ' in command and 'agentweave' in command
        if (is_broker or is_worker) and (base in (info['cwd'] or '') or base in command):
            found.append(process)
    return found


@pytest.fixture(autouse=True)
def no_leaked_processes(tmp_path_factory):
    """Fail the test that leaves its own broker or worker running, then stop them."""
    import time
    import psutil
    base = str(tmp_path_factory.getbasetemp())
    before = {process.pid for process in _test_processes(base)}
    yield
    deadline = time.monotonic() + LEAK_GRACE_SECONDS
    leaked = [p for p in _test_processes(base) if p.pid not in before]
    while leaked and time.monotonic() < deadline:
        time.sleep(.1)
        leaked = [p for p in leaked if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
    if not leaked:
        return
    described = [f'{p.pid}: {" ".join(p.cmdline())[:160]}' for p in leaked]
    for process in leaked:
        process.terminate()
    _, alive = psutil.wait_procs(leaked, timeout=15)
    for process in alive:
        process.kill()
    pytest.fail('test left processes running: ' + '; '.join(described))
