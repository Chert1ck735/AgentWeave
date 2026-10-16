"""The load driver at small scale: real CLI workers, journal accounting and a complete report."""
from dataclasses import asdict
import pytest


def test_a_short_load_run_loses_nothing_and_reports_every_measurement(tmp_path):
    from scripts.load_test import run_load
    root = tmp_path / 'load project'
    root.mkdir()
    report = run_load('dev', agents=2, rate=20, seconds=3, payload_bytes=1024, request_rate=5, root=root)
    assert report.published == 60 and report.publish_errors == 0
    assert (report.accepted, report.completed, report.lost, report.not_admitted) == (60, 60, 0, 0)
    assert report.drain_seconds is not None and report.queue_shrinking_after_publish
    assert report.requests == 15 and report.request_errors == 0 and report.request_p95_seconds is not None
    assert report.peak_rss_bytes > 0 and report.environment['python'] and report.baseline['memory_total_bytes']
    assert set(report.checks) == {'every_event_published', 'no_lost_accepted_work', 'accepted_work_completed',
                                  'every_published_event_admitted', 'no_request_errors', 'queue_drained',
                                  'queue_shrinks_after_publishing', 'request_p95_within_500ms',
                                  'extra_ram_within_4gib', 'target_rate_reached'}
    assert all(report.checks[name] for name in ('every_event_published', 'no_lost_accepted_work',
                                                'accepted_work_completed', 'every_published_event_admitted',
                                                'no_request_errors', 'queue_drained'))
    assert asdict(report)['provider_attempts'] == 0  # handlers only, no paid model


def test_only_the_dev_profile_is_driven_in_v01():
    from scripts.load_test import run_load
    with pytest.raises(ValueError, match='v0.2'):
        run_load('server')
