"""Real owned broker storage lifecycle, separate from the permanent SDK journal."""
import os
from pathlib import Path
import socket
import stat
from uuid import uuid4
import pytest


def child_directory(broker):
    return Path(os.readlink(f'/proc/{broker.process.pid}/cwd'))


def sqlite_files(directory):
    result = []
    for path in directory.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                if stream.read(16) == b'SQLite format 3\x00':
                    result.append(path)
    return result


@pytest.mark.integration
async def test_restart_removes_only_old_owned_storage_and_keeps_sdk_journal(broker_process, store_factory, envelope_factory):
    from confluent_kafka.admin import AdminClient, NewTopic
    broker = broker_process
    old_directory = child_directory(broker)
    assert old_directory != broker.work_dir
    assert stat.S_IMODE(old_directory.stat().st_mode) == 0o700
    assert sqlite_files(old_directory)
    sentinel = broker.work_dir / 'unrelated.txt'
    sentinel.write_text('keep this unrelated file')
    sdk = await store_factory(broker.work_dir / 'sdk-journal.sqlite3')
    event = envelope_factory()
    await sdk.admit(event, 'v1')
    name = 'aw.lifecycle.' + uuid4().hex
    admin = AdminClient({'bootstrap.servers':broker.url, 'log_level':0})
    admin.create_topics([NewTopic(name, num_partitions=1, replication_factor=1)])[name].result(10)
    old_epoch, address, pid = broker.epoch, broker.url, broker.process.pid
    with pytest.raises(RuntimeError):
        broker.start()
    assert broker.process.pid == pid and broker.process.poll() is None
    assert name in admin.list_topics(timeout=6).topics
    broker.stop()
    assert not old_directory.exists()
    assert sentinel.read_text() == 'keep this unrelated file'
    assert sdk.path.exists()
    await sdk.close()
    reopened = await store_factory(sdk.path)
    assert (await reopened.get_task(event.task_id)).definition_hash == 'v1'
    broker.start()
    fresh_directory = child_directory(broker)
    assert fresh_directory != old_directory
    assert sqlite_files(fresh_directory)
    assert broker.epoch != old_epoch and broker.url == address
    assert name not in admin.list_topics(timeout=6).topics
    broker.stop()
    assert not fresh_directory.exists()
    assert sentinel.read_text() == 'keep this unrelated file'
    assert (await reopened.get_task(event.task_id)).status == 'accepted'


@pytest.mark.integration
def test_failed_real_broker_start_cleans_its_owned_storage(tmp_path):
    from agentweave_dev import resolve_broker_bin
    from scripts.probe_broker import BrokerProcess
    directory = tmp_path / 'failed start with spaces'
    directory.mkdir()
    sentinel = directory / 'unrelated.txt'
    sentinel.write_text('keep on failure')
    broker = BrokerProcess(resolve_broker_bin(), directory, 'aw-failed-' + uuid4().hex)
    try:
        with socket.socket() as foreign_listener:
            foreign_listener.bind(('127.0.0.1', int(broker.url.rsplit(':', 1)[1])))
            foreign_listener.listen(1)
            with pytest.raises(RuntimeError):
                broker.start()
            assert broker.process.poll() is not None
            assert sentinel.read_text() == 'keep on failure'
            assert sorted(path.name for path in directory.iterdir()) == ['broker.log', 'unrelated.txt']
            assert broker.log is None
    finally:
        broker.stop()


@pytest.mark.integration
def test_private_backend_handles_unicode_spaces_and_symlink_workdir(tmp_path):
    from agentweave_dev import resolve_broker_bin
    from scripts.probe_broker import BrokerProcess
    real = tmp_path / 'проект with spaces'
    real.mkdir()
    alias = tmp_path / 'chosen project'
    alias.symlink_to(real, target_is_directory=True)
    broker = BrokerProcess(resolve_broker_bin(), alias, 'aw-path-' + uuid4().hex)
    try:
        broker.start()
        directory = child_directory(broker)
        assert directory.is_relative_to(real)
        assert directory != real
        assert sqlite_files(directory)
        broker.stop()
        assert not directory.exists()
        assert alias.is_symlink() and real.is_dir()
    finally:
        broker.stop()


@pytest.mark.integration
def test_startup_failure_before_popen_preserves_foreign_log_directory(tmp_path):
    from agentweave_dev import resolve_broker_bin
    from scripts.probe_broker import BrokerProcess
    foreign = tmp_path / 'broker.log'
    foreign.mkdir()
    sentinel = foreign / 'unrelated.txt'
    sentinel.write_text('keep foreign directory')
    broker = BrokerProcess(resolve_broker_bin(), tmp_path, 'aw-log-' + uuid4().hex)
    try:
        with pytest.raises(IsADirectoryError):
            broker.start()
        assert broker.process is None
        assert set(tmp_path.iterdir()) == {foreign}
        assert sentinel.read_text() == 'keep foreign directory'
    finally:
        broker.stop()
