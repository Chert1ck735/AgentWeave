"""Low-cardinality runtime metrics and secret-free correlation logs."""
import logging
from collections import Counter


class RuntimeMetrics:
    def __init__(self):
        self.counters = Counter()
        self.queues = {}
        self.result_latency = []

    def queue(self, subscription, depth, age):
        self.queues[subscription] = {'depth':depth, 'age_seconds':max(0, age)}

    def outcome(self, name, *, event=None, task_id=None, instance_id=None):
        self.counters[name] += 1
        logging.getLogger('agentweave.runtime').info(name, extra={
            'network_id':event.network_id if event else None,
            'event_id':event.event_id if event else None,
            'task_id':task_id, 'instance_id':instance_id})

    def latency(self, seconds):
        # Fixed-size diagnostic reservoir; labels never contain event/task IDs.
        self.result_latency.append(max(0, seconds))
        del self.result_latency[:-64]
