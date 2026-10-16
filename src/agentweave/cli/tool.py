"""Published tool services: list and describe live tools, stop this project's own processes."""
import asyncio
from .._infrastructure.processes import InfrastructureError

SETTLE_SECONDS = 0.3
OWNED_WAIT_SECONDS = 5.0


def _owned(records, tool_id):
    return any(r.logical_id == tool_id or tool_id.startswith(r.logical_id + '.') for r in records)


async def _catalog(root, profile):
    """Live tool announcements and this project's tool service records.

    The directory is the only liveness authority; the journal never stands in for it.
    """
    from ..config import resolve
    from ..errors import ConfigError
    from .._infrastructure.bootstrap import Bootstrap
    from .agent import is_tool_record, select_records
    _, records = select_records(root, profile)
    records = [r for r in records if is_tool_record(r)]
    try:
        config = resolve(root, profile)
    except ConfigError:
        return {}, records
    if not (config.broker_url and config.database_url and config.broker_epoch):
        return {}, records
    resources = await Bootstrap.open_resources(config)
    try:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(SETTLE_SECONDS)  # fold retained announcements
        deadline = loop.time() + OWNED_WAIT_SECONDS
        while loop.time() < deadline:
            visible = resources.directory.snapshot()
            if all(any(_owned([r], tool_id) for tool_id in visible) for r in records):
                break
            await asyncio.sleep(.05)
        snapshot = {tool_id: instances for tool_id, instances in resources.directory.snapshot().items()
                    if instances[0].kind == 'tool'}
    finally:
        await resources.close()
    return snapshot, records


def _entry(tool_id, instances, records):
    first = instances[0]
    return {'tool_id': tool_id, 'name': first.name, 'revision': first.revision,
            'description': first.description, 'instances': len(instances),
            'allowed_callers': first.allowed_callers, 'owned': _owned(records, tool_id)}


def list_tools(root, profile):
    snapshot, records = asyncio.run(_catalog(root, profile))
    return {'tools': [_entry(tool_id, instances, records) for tool_id, instances in sorted(snapshot.items())]}


def describe(root, profile, name):
    snapshot, records = asyncio.run(_catalog(root, profile))
    if name not in snapshot:
        raise InfrastructureError('tool_not_found', 'no live tool is published under that name')
    first = snapshot[name][0]
    return _entry(name, snapshot[name], records) | {'input_schema': first.input_schema,
                                                     'output_schema': first.output_schema}


def stop(root, profile, name, instance=None):
    """Stop this project's tool service processes registered as name; never an agent or a foreign process."""
    from .agent import is_tool_record, select_records, stop_workers
    _, records = select_records(root, profile, name)
    if not any(is_tool_record(r) for r in records):
        raise InfrastructureError('tool_not_found', 'no tool service of this project runs under that name')
    return stop_workers(root, profile, name, instance, only=is_tool_record)
