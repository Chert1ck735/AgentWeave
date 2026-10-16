"""Real protocol acceptance runner for an explicitly supplied local broker binary."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import confluent_kafka as ck
from confluent_kafka.admin import AdminClient, NewTopic


class BrokerProcess:
    """Own a broker child and disposable SQLite storage for each start epoch."""

    def __init__(self, binary: Path, work_dir: Path, network_id: str):
        self.binary = Path(binary).resolve(strict=True)
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.network_id = network_id
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.url = f'127.0.0.1:{sock.getsockname()[1]}'
        self.process = None
        self.log = None
        self.epoch = ''
        self._epoch_storage = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError('Broker child is already running')
        # A child may have exited without an explicit stop. Clean only this
        # instance's previous storage/log before allocating a fresh epoch.
        self.stop()
        self.epoch = ''
        try:
            self._epoch_storage = tempfile.TemporaryDirectory(
                prefix='agentweave-broker-epoch-', dir=self.work_dir)
            self.log = (self.work_dir / 'broker.log').open('ab')
            args = [str(self.binary), '--silent', '--cluster-id', self.network_id,
                    '--listener-url', f'tcp://{self.url}',
                    '--advertised-listener-url', f'tcp://{self.url}',
                    '--storage-engine', 'sqlite:///broker.sqlite']
            env = {k: v for k, v in os.environ.items() if k in ('PATH', 'LANG', 'LC_ALL')}
            env['RUST_LOG'] = 'warn'
            # Relative SQLite URI resolution is verified for this binary. A
            # private cwd avoids encoding a user path into the backend URI.
            self.process = subprocess.Popen(args, cwd=self._epoch_storage.name, env=env,
                                            stdout=self.log, stderr=self.log)
            admin = AdminClient({'bootstrap.servers': self.url, 'socket.timeout.ms': 6000, 'log_level':0})
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Broker exited {self.process.returncode}; see {self.work_dir / "broker.log"}')
                try:
                    admin.list_topics(timeout=1)
                    self.epoch = uuid.uuid4().hex
                    return self.url
                except ck.KafkaException:
                    time.sleep(.1)
            raise TimeoutError('Broker readiness timed out')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        # Popen owns the child; never identify or signal an unrelated PID.
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(5)
        if self.log is not None:
            self.log.close()
            self.log = None
        if self._epoch_storage is not None:
            self._epoch_storage.cleanup()
            self._epoch_storage = None

    def restart(self) -> str:
        self.stop()
        self.start()
        return self.epoch


def check_protocol(broker: BrokerProcess) -> dict[str, bool]:
    """Exercise the seven operations against a running real broker."""
    checks = dict.fromkeys(('create', 'keyed', 'two_groups', 'rebalance', 'commit', 'reconnect', 'payload_1mib'), False)
    consumers = []
    base = {'bootstrap.servers': broker.url, 'socket.timeout.ms': 6000}
    admin = AdminClient(base)
    producer = ck.Producer(dict(base, **{'enable.idempotence': False,
        'message.max.bytes': 2097152, 'compression.type': 'none', 'delivery.timeout.ms': 15000}))

    def announce(name):
        checks[name] = True

    def topic(name, partitions=2):
        futures = admin.create_topics([NewTopic(name, num_partitions=partitions, replication_factor=1)], request_timeout=6)
        futures[name].result(10)
        assert len(admin.list_topics(name, timeout=6).topics[name].partitions) == partitions

    def send(name, key, value, partition=None):
        results = []
        kwargs = {'key': key, 'value': value, 'on_delivery': lambda e, m: results.append((e, m))}
        if partition is not None:
            kwargs['partition'] = partition
        producer.produce(name, **kwargs)
        assert producer.flush(20) == 0
        assert len(results) == 1 and results[0][0] is None, repr(results)
        return results[0][1]

    def consumer(group, name):
        c = ck.Consumer(dict(base, **{'group.id': group, 'group.protocol': 'classic',
           'enable.auto.commit': False, 'enable.auto.offset.store': True,
           'auto.offset.reset': 'earliest', 'session.timeout.ms': 6000,
           'heartbeat.interval.ms': 1000, 'max.poll.interval.ms': 60000,
           'partition.assignment.strategy': 'range', 'fetch.message.max.bytes': 2097152}))
        consumers.append(c)
        c.subscribe([name])
        return c

    def receive(c, expected, timeout=30):
        data = []
        end = time.monotonic() + timeout
        while len(data) < expected and time.monotonic() < end:
            m = c.poll(.2)
            if m is None:
                continue
            if m.error():
                raise ck.KafkaException(m.error())
            data.append(m)
        assert len(data) == expected, f'received {len(data)}/{expected}; assignment={c.assignment()}'
        return data

    started = time.monotonic()
    errors = {}
    failed_stage = 'create'
    try:
        name = 'aw.probe.' + uuid.uuid4().hex
        failed_stage = 'create'
        topic(name)
        announce('create')
        failed_stage = 'keyed'
        first = send(name, b'session-1', b'one')
        second = send(name, b'session-1', b'two')
        assert first.partition() == second.partition()
        group_a = 'aw-group-a-' + uuid.uuid4().hex
        a = consumer(group_a, name)
        records_a = receive(a, 2)
        assert [m.value() for m in records_a] == [b'one', b'two']
        assert all(m.key() == b'session-1' for m in records_a)
        announce('keyed')
        failed_stage = 'two_groups'
        b = consumer('aw-group-b-' + uuid.uuid4().hex, name)
        records_b = receive(b, 2)
        assert [m.value() for m in records_b] == [b'one', b'two']
        announce('two_groups')
        failed_stage = 'commit'
        committed = a.commit(asynchronous=False)
        actual = a.committed([ck.TopicPartition(name, first.partition())], timeout=6)
        assert actual[0].offset == second.offset() + 1, actual
        a.close(); consumers.remove(a)
        a = consumer(group_a, name)
        # The same group must resume after the saved offset, without replaying old records.
        send(name, b'session-1', b'after-commit')
        resumed = receive(a, 1)
        assert resumed[0].value() == b'after-commit', 'manual offset was not respected on reopen'
        announce('commit')
        failed_stage = 'rebalance'
        b.close(); consumers.remove(b)
        rebalance_group = 'aw-rebalance-' + uuid.uuid4().hex
        r1 = consumer(rebalance_group, name)
        end = time.monotonic() + 30
        while time.monotonic() < end and len(r1.assignment()) != 2:
            r1.poll(.2)
        assert len(r1.assignment()) == 2, r1.assignment()
        r2 = consumer(rebalance_group, name)
        end = time.monotonic() + 35
        while time.monotonic() < end:
            r1.poll(.1); r2.poll(.1)
            x = {p.partition for p in r1.assignment()}
            y = {p.partition for p in r2.assignment()}
            if len(x) == len(y) == 1 and not (x & y) and x | y == {0, 1}:
                break
        else:
            raise AssertionError(f'rebalance assignments {r1.assignment()} / {r2.assignment()}')
        send(name, b'partition-0', b'rebalanced-0', partition=0)
        send(name, b'partition-1', b'rebalanced-1', partition=1)
        got = set()
        end = time.monotonic() + 20
        while time.monotonic() < end and got != {b'rebalanced-0', b'rebalanced-1'}:
            for c in (r1, r2):
                m = c.poll(.1)
                if m and m.error():
                    raise ck.KafkaException(m.error())
                if m and m.value().startswith(b'rebalanced-'):
                    got.add(m.value())
        assert got == {b'rebalanced-0', b'rebalanced-1'}, got
        announce('rebalance')
        failed_stage = 'payload_1mib'
        big = 'aw.big.' + uuid.uuid4().hex
        topic(big, 1)
        payload = b'x' * 1048576
        send(big, b'large', payload, 0)
        large_c = consumer('aw-large-' + uuid.uuid4().hex, big)
        large = receive(large_c, 1)[0]
        assert len(large.value()) == 1048576 and large.value() == payload
        announce('payload_1mib')
        failed_stage = 'reconnect'
        for c in list(consumers):
            c.close(); consumers.remove(c)
        old_epoch = broker.epoch
        assert broker.restart() != old_epoch
        assert name not in admin.list_topics(timeout=6).topics, 'prior epoch topics unexpectedly survived restart'
        restarted = 'aw.restarted.' + uuid.uuid4().hex
        topic(restarted, 1)
        # Keep the original producer/admin across the restart to test connection recovery.
        send(restarted, b'after', b'recovered', 0)
        c = consumer('aw-after-' + uuid.uuid4().hex, restarted)
        assert receive(c, 1)[0].value() == b'recovered'
        announce('reconnect')
    except Exception as exc:
        errors[failed_stage] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for consumer in consumers:
            consumer.close()
        report = {'checks': checks, 'errors': errors,
                  'duration_seconds': round(time.monotonic() - started, 3),
                  'binary_sha256': hashlib.sha256(broker.binary.read_bytes()).hexdigest(),
                  'python': sys.version.split()[0], 'platform': platform.platform(),
                  'confluent_kafka': version('confluent-kafka'),
                  'binding_version': ck.version(), 'librdkafka': ck.libversion()[0],
                  'storage': 'ephemeral_sqlite', 'network_id': broker.network_id}
        (broker.work_dir / 'probe-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return checks


def probe_broker(binary: Path, *, work_dir: Path | None = None) -> dict[str, bool]:
    """Start, exercise, restart and stop only this invocation's broker child."""
    with tempfile.TemporaryDirectory(prefix='agentweave-probe-') as temporary:
        broker = BrokerProcess(binary, work_dir or Path(temporary), 'aw-probe-' + uuid.uuid4().hex)
        try:
            broker.start()
            return check_protocol(broker)
        finally:
            broker.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--work-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(probe_broker(args.binary, work_dir=args.work_dir), indent=2))
