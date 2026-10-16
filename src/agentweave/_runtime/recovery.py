"""Bounded journal recovery; never relies on an earlier broker inbox."""
import asyncio

# Recovery is woken by the engine when local work frees up; this fallback finds work freed
# by other processes (another replica, an operator, a crashed owner's expired lease).
RECOVERY_FALLBACK_SECONDS = 1.0


async def recover(worker):
    while not worker.stopping:
        worker.recovery_wake.clear()  # a wake during the scan below triggers the next one at once
        cursor = None
        while not worker.stopping:
            page = await worker.store.recoverable_page(worker.definition.name,
                after_task_id=cursor, limit=64)
            if not page:
                break
            for task in page:
                if task.definition_hash != worker.fingerprint:
                    raise ValueError('accepted task definition fingerprint differs from Worker')
            # Each batch has at most the logical execution limit. The engine and
            # store independently enforce local slots and replica-wide leases.
            # outcome_unknown waits for an operator; executing it only costs a write.
            for task in page:
                if task.status == 'waiting_peer':
                    # A lost reply must not strand a waiting loop; a dead waiting handler is
                    # reclaimed; a wait nobody answers ends at its deadline.
                    if not await worker.coordinator.resume_finished(task):
                        await worker.store.reclaim_handler_wait(task.task_id)
                        await worker.store.expire_task(task.task_id)
            page = [await worker.store.get_task(task.task_id) for task in page]
            runnable = [task for task in page if task.status not in ('outcome_unknown', 'waiting_peer')]
            width = worker.definition.concurrency
            for offset in range(0, len(runnable), width):
                await asyncio.gather(*(worker.engine.execute(task.task_id)
                    for task in runnable[offset:offset+width]))
            cursor = page[-1].task_id
        try:
            await asyncio.wait_for(worker.recovery_wake.wait(), RECOVERY_FALLBACK_SECONDS)
        except TimeoutError:
            pass
