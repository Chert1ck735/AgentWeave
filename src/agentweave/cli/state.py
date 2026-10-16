"""state inspect/export/cleanup: diagnostics and retention of the profile journal."""
from pathlib import Path


def command(root, profile, args, emit):
    """Dispatch one state command; returns the exit code."""
    from .task import operator_errors
    return operator_errors(_command(root, profile, args, emit))


async def _command(root, profile, args, emit):
    from .._runtime import maintenance
    from .event import session
    from .task import journal
    session_id = session(args.session_id) if args.action == 'inspect' else None
    async with journal(root, profile) as store:
        if args.action == 'inspect':
            emit(await maintenance.inspect_state(store, session_id))
        elif args.action == 'export':
            emit(await maintenance.export_state(store, Path(args.path).resolve()))
        else:
            # A dry run unless --apply; either way one serialized journal transaction.
            emit(await maintenance.cleanup(store, dry_run=not args.apply))
    return 0
