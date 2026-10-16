"""Load driver for the dev profile: real CLI workers, real broker, journal-based accounting.

Agents are started with the CLI from a temporary project. Events are published
at a fixed rate (each agent subscribes to its own topic) while a request/reply
stream measures latency. Accounting comes from the journal, not from worker
memory. An admitted task that never reaches a terminal status is lost; an
acknowledged event that never became a task is not_admitted (overload, or input
that expired before admission), which is reported separately, never as success.

    python scripts/load_test.py --profile dev --agents 10 --rate 100 --seconds 60 --payload-bytes 1024
"""
import argparse
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
import platform
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
DRAIN_SECONDS = 120
SAMPLE_SECONDS = 1.0
P95_TARGET_SECONDS = 0.5
EXTRA_RAM_TARGET_BYTES = 4 * 1024 ** 3
RATE_TOLERANCE = 0.95
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_AGENT = 'loadrr'

APP = '''from agentweave import Agent
AGENTS = {agents}
for index in range(AGENTS):
    agent = Agent(f'load{{index}}', max_inflight=4096)

    @agent.on(f'load.a{{index}}', retry='safe')
    async def work(ctx, event):
        return len(event['data'])

    @agent.on_request
    async def echo(ctx, request):
        return request.input
    globals()[f'load{{index}}'] = agent

# Request/reply is measured on its own agent, beside the event load: one agent serving both
# would measure that agent's saturation, not request/reply latency under load.
loadrr = Agent('loadrr')


@loadrr.on_request
async def answer(ctx, request):
    return request.input
'''


@dataclass
class LoadReport:
    profile: str
    agents: int
    target_rate: float
    seconds: float
    payload_bytes: int
    published: int = 0
    publish_errors: int = 0
    achieved_rate: float = 0.0
    accepted: int = 0
    completed: int = 0
    failed: int = 0
    lost: int = 0
    not_admitted: int = 0
    expired: int = 0
    unknown: int = 0
    max_queue_depth: int = 0
    max_queue_age_seconds: float = 0.0
    queue_shrinking_after_publish: bool = False
    drain_seconds: float | None = None
    requests: int = 0
    request_errors: int = 0
    request_error_kinds: dict = field(default_factory=dict)
    request_p50_seconds: float | None = None
    request_p95_seconds: float | None = None
    provider_attempts: int = 0
    provider_tokens: int = 0
    provider_cost_usd: str = '0'
    provider_unknown: int = 0
    peak_rss_bytes: int = 0
    baseline: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    def evaluate(self) -> dict:
        """The v0.1 criteria, each PASS or FAIL against its measured value; nothing is adjusted to pass."""
        self.checks = {
            'every_event_published': self.publish_errors == 0,
            'no_lost_accepted_work': self.lost == 0,
            'accepted_work_completed': self.completed == self.accepted,  # expired or unknown work is not done
            'every_published_event_admitted': self.not_admitted == 0,
            'no_request_errors': self.request_errors == 0,
            'queue_drained': self.drain_seconds is not None,
            'queue_shrinks_after_publishing': self.queue_shrinking_after_publish,
            'request_p95_within_500ms': (self.request_p95_seconds is not None
                                         and self.request_p95_seconds <= P95_TARGET_SECONDS),
            'extra_ram_within_4gib': self.peak_rss_bytes <= EXTRA_RAM_TARGET_BYTES,
            'target_rate_reached': self.achieved_rate >= RATE_TOLERANCE * self.target_rate,
        }
        return self.checks


def _environment():
    import importlib.metadata
    versions = {}
    for package in ('agentweave', 'agentweave-dev', 'confluent-kafka', 'pydantic'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'source'
    try:
        from agentweave._infrastructure.local import packaged_binary
        versions['broker'] = packaged_binary()[1]
    except Exception as error:  # the report says why it is missing instead of failing the run
        versions['broker'] = f'unavailable: {type(error).__name__}'
    return {'python': platform.python_version(), 'platform': platform.platform(), 'cpus': os.cpu_count(), **versions}


def _baseline():
    import psutil
    memory = psutil.virtual_memory()
    return {'load_average': os.getloadavg(), 'memory_total_bytes': memory.total,
            'memory_available_bytes': memory.available}


def _env():
    paths = [str(REPO / 'src'), str(REPO / 'packages' / 'agentweave-dev' / 'src'), os.environ.get('PYTHONPATH', '')]
    return os.environ | {'PYTHONPATH': os.pathsep.join(path for path in paths if path)}


def _cli(root, profile, *args, timeout=90):
    result = subprocess.run([sys.executable, '-m', 'agentweave', '--root', str(root), '--profile', profile, '--json',
                             *args], cwd=REPO, env=_env(), capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f'agentweave {" ".join(args)} failed: {result.stdout.strip()[-500:]}')
    return json.loads(result.stdout.splitlines()[-1])


def _rss(root, profile):
    """Resident memory of this driver plus every worker and broker the project registry owns."""
    import psutil
    from agentweave._infrastructure.processes import ProcessRegistry
    total = psutil.Process().memory_info().rss
    for record in ProcessRegistry(root, profile).records():
        try:
            total += psutil.Process(record.pid).memory_info().rss
        except psutil.Error:
            pass
    return total


def _journal(root, profile, sql, *values):
    from agentweave.config import resolve
    path = resolve(root, profile).database_url.removeprefix('sqlite:///')
    with sqlite3.connect(path, timeout=30) as connection:
        return connection.execute(sql, values).fetchall()


# Tasks of this run only: a reused --root may hold the journal of earlier runs.
_RUN_TASKS = "json_extract(input_envelope,'$.topic') LIKE 'load.%' AND admitted_at>=?"


def _queue(root, profile, now, since):
    """Admitted, unfinished work in the journal; events still waiting in the broker are not visible here."""
    [(depth, oldest)] = _journal(root, profile, "SELECT COUNT(*), MIN(admitted_at) FROM tasks WHERE status IN "
                                 f"('accepted','running','waiting_tool') AND {_RUN_TASKS}", since)
    return depth, 0.0 if not oldest else max(0.0, now - datetime.fromisoformat(oldest).timestamp())


async def _publish(root, profile, agents, rate, seconds, payload):
    """Publish at the target rate; returns (published, errors, achieved rate)."""
    from agentweave import Client
    published = errors = 0
    async with Client.connect(root, profile) as client:
        started = time.monotonic()
        for index in range(int(rate * seconds)):
            delay = started + index / rate - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                await client.publish(f'load.a{index % agents}', {'data': payload, 'n': index})
                published += 1
            except Exception:
                errors += 1
        elapsed = max(time.monotonic() - started, 1e-9)
    return published, errors, round(published / elapsed, 2)


def _publisher(root, profile, agents, rate, seconds, payload, results):
    """The publisher's own process: waiting requests in the driver cannot slow the event stream."""
    try:
        results.put(('done', asyncio.run(_publish(root, profile, agents, rate, seconds, payload))))
    except Exception as error:
        results.put(('error', f'{type(error).__name__}: {error}'))


async def _requests(client, rate, seconds, latencies, errors):
    """Each request's latency; a request that fails counts as REQUEST_TIMEOUT_SECONDS in the percentiles."""
    started, pending = time.monotonic(), []

    async def one(index):
        begun = time.monotonic()
        try:
            result = await asyncio.wait_for(client.agent(REQUEST_AGENT).ask({'ping': index}), REQUEST_TIMEOUT_SECONDS)
            if result.status == 'succeeded':
                latencies.append(time.monotonic() - begun)
                return
            errors.append(result.status)
        except Exception as error:
            errors.append(type(error).__name__)
        latencies.append(float(REQUEST_TIMEOUT_SECONDS))  # never survivorship: a failure is the worst latency
    for index in range(int(rate * seconds)):
        delay = started + index / rate - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        pending.append(asyncio.create_task(one(index)))
    await asyncio.gather(*pending)


async def _sample(root, profile, report, samples, stop, since):
    while not stop.is_set():
        depth, age = await asyncio.to_thread(_queue, root, profile, time.time(), since)
        samples.append(depth)
        report.max_queue_depth = max(report.max_queue_depth, depth)
        report.max_queue_age_seconds = round(max(report.max_queue_age_seconds, age), 3)
        report.peak_rss_bytes = max(report.peak_rss_bytes, await asyncio.to_thread(_rss, root, profile))
        try:
            await asyncio.wait_for(stop.wait(), SAMPLE_SECONDS)
        except TimeoutError:
            pass


def _account(root, profile, report, since):
    counts = dict(_journal(root, profile, f"SELECT status, COUNT(*) FROM tasks WHERE {_RUN_TASKS} GROUP BY status",
                           since))
    report.accepted = sum(counts.values())
    report.completed = counts.get('succeeded', 0)
    report.expired = counts.get('expired', 0)
    report.unknown = counts.get('outcome_unknown', 0)
    report.failed = counts.get('failed', 0) + counts.get('cancelled', 0)
    terminal = report.completed + report.failed + report.unknown + report.expired
    report.lost = report.accepted - terminal  # admitted work that is not finished (yet)
    report.not_admitted = report.published - report.accepted
    [(attempts, tokens, unknown)] = _journal(root, profile, 'SELECT COUNT(*), COALESCE(SUM(input_tokens+output_tokens),0),'
                                             ' SUM(CASE WHEN usage_known=0 THEN 1 ELSE 0 END) FROM model_attempts')
    report.provider_attempts, report.provider_tokens, report.provider_unknown = attempts, tokens, unknown or 0


def _settled(report):
    return report.not_admitted <= 0 and report.lost <= 0


async def _drive(root, profile, report, agents, rate, seconds, payload_bytes, request_rate):
    from agentweave import Client
    payload = 'x' * max(0, payload_bytes - 32)  # the event's JSON framing adds the rest
    latencies, errors, samples, stop = [], [], [], asyncio.Event()
    from datetime import timezone
    since = datetime.now(timezone.utc).isoformat(timespec='microseconds')
    async with Client.connect(root, profile) as client:
        for name in [f'load{index}' for index in range(agents)] + [REQUEST_AGENT]:
            async with asyncio.timeout(30):
                while not client.directory.snapshot().get(name):
                    await asyncio.sleep(0.05)
        sampler = asyncio.create_task(_sample(root, profile, report, samples, stop, since))
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        results = context.Queue()
        publisher = context.Process(target=_publisher, args=(str(root), profile, agents, rate, seconds, payload, results))
        publisher.start()
        try:
            await _requests(client, request_rate, seconds, latencies, errors)
            outcome = await asyncio.to_thread(results.get, True, seconds + 120)
            await asyncio.to_thread(publisher.join, 30)
        finally:
            if publisher.is_alive():
                publisher.terminate()
                await asyncio.to_thread(publisher.join, 10)
        if outcome[0] != 'done':
            raise RuntimeError(f'the publisher failed: {outcome[1]}')
        report.published, report.publish_errors, report.achieved_rate = outcome[1]
        published_at = len(samples)
        drain_started = time.monotonic()
        while time.monotonic() < drain_started + DRAIN_SECONDS:
            await asyncio.to_thread(_account, root, profile, report, since)
            if _settled(report):
                report.drain_seconds = round(time.monotonic() - drain_started, 3)
                break
            await asyncio.sleep(0.5)
        stop.set()
        await sampler
    after = samples[published_at:]
    report.queue_shrinking_after_publish = report.drain_seconds is not None and (not after or after[-1] <= after[0])
    report.requests, report.request_errors = len(latencies), len(errors)
    report.request_error_kinds = {kind: errors.count(kind) for kind in sorted(set(errors))}
    if latencies:
        ordered = sorted(latencies)
        report.request_p50_seconds = round(statistics.median(ordered), 4)
        report.request_p95_seconds = round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 4)
    await asyncio.to_thread(_account, root, profile, report, since)


def run_load(profile='dev', agents=10, rate=100, seconds=60, payload_bytes=1024, *, request_rate=10,
             root=None) -> LoadReport:
    """Run the load and return its report; workers and the broker are stopped before returning."""
    if profile != 'dev':
        raise ValueError('the load driver runs the dev profile in v0.1; the server profile comes with v0.2')
    report = LoadReport(profile=profile, agents=agents, target_rate=rate, seconds=seconds, payload_bytes=payload_bytes,
                        baseline=_baseline(), environment=_environment())
    directory = Path(tempfile.mkdtemp(prefix='agentweave-load-')) if root is None else Path(root)
    (directory / 'app.py').write_text(APP.format(agents=agents))
    started = []
    try:
        for name in [f'load{index}' for index in range(agents)] + [REQUEST_AGENT]:
            _cli(directory, profile, 'agent', 'start', f'app:{name}')
            started.append(name)
        asyncio.run(_drive(directory, profile, report, agents, rate, seconds, payload_bytes, request_rate))
    finally:
        for name in started:
            try:
                _cli(directory, profile, 'agent', 'stop', name)
            except Exception as error:
                print(f'warning: stopping {name} failed: {error}', file=sys.stderr)
        try:
            _cli(directory, profile, 'infra', 'down')
        except Exception as error:
            print(f'warning: infra down failed: {error}', file=sys.stderr)
    report.evaluate()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--profile', default='dev', choices=('dev', 'server'))
    parser.add_argument('--agents', type=int, default=10)
    parser.add_argument('--rate', type=float, default=100)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--payload-bytes', type=int, default=1024)
    parser.add_argument('--request-rate', type=float, default=10)
    parser.add_argument('--root', help='project directory to use; a temporary one by default')
    parser.add_argument('--json-out', help='also write the report to this file')
    args = parser.parse_args(argv)
    report = run_load(args.profile, args.agents, args.rate, args.seconds, args.payload_bytes,
                      request_rate=args.request_rate, root=args.root)
    text = json.dumps(asdict(report), indent=2, default=str)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + '\n')
    return 0 if all(report.checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
