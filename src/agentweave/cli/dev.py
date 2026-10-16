"""One-command local agent launch and optional request loop."""
import asyncio
import json
import signal
from .._infrastructure.bootstrap import Bootstrap
from .agent import load_definition,start_worker,supervise,stop_workers


def _interrupt(signum,frame):
    raise KeyboardInterrupt


def run(target,root,profile,*,detach=False,chat=False,replica=False,overrides=None,emit=lambda value:None,stage=lambda value:None):
    stage('validate definition')
    load_definition(target,root,overrides)
    stage('prepare profile, journal and broker')
    config = Bootstrap(root,profile).prepare()
    stage('wait for worker heartbeat')
    if detach:
        record = start_worker(target,config,detach,replica=replica,overrides=overrides)
    else:
        # A foreground parent owns its child from spawn on: a signal during startup
        # becomes an interrupt, so start_worker's cleanup stops the child before exit.
        previous = {sig:signal.signal(sig,_interrupt) for sig in (signal.SIGTERM,signal.SIGHUP)}
        try:
            record = start_worker(target,config,detach,replica=replica,overrides=overrides)
        finally:
            for sig,handler in previous.items(): signal.signal(sig,handler)
    from dataclasses import asdict
    emit(asdict(record)|{'ready':True,'connect':f'Client.connect(root={str(root)!r}, profile={profile!r})'})
    if chat:
        try: asyncio.run(request_loop(root,profile,record.logical_id))
        finally: stop_workers(root,profile,record.logical_id,record.instance_id)
    elif not detach:
        supervise(record,root,profile)
    return record


AGENT_WAIT_SECONDS = 20


async def wait_for_agent(client,name):
    """A fresh client learns the directory first; an agent that never appears is unavailable."""
    from .._infrastructure.processes import InfrastructureError
    try:
        async with asyncio.timeout(AGENT_WAIT_SECONDS):
            while not client.directory.snapshot().get(name): await asyncio.sleep(.05)
    except TimeoutError as exc:
        raise InfrastructureError('agent_unavailable',f'agent {name} is not running') from exc


async def request(root,profile,name,payload,*,session_id=None,wait=True,emit):
    """One request: acceptance first (the task_id), then the terminal result.

    Returns the exit code: 0 when the task succeeded, 4 when it failed or its outcome is unknown.
    """
    from ..client import Client
    from ..errors import DefinitionError
    from .._infrastructure.processes import InfrastructureError
    from .infra import require_running
    require_running(root,profile)
    async with Client.connect(root,profile) as client:
        await wait_for_agent(client,name)
        try:
            submission = await client.agent(name).submit(payload,session_id)
        except DefinitionError as exc:
            raise InfrastructureError('agent_unavailable',f'agent {name} cannot accept requests') from exc
        emit({'task_id':submission.task_id,'accepted':True})
        if not wait:
            return 0
        result = await client.result(submission.task_id)  # terminal by the task deadline at the latest
        emit(result.model_dump(mode='json'))
        return 0 if result.status == 'succeeded' else 4


async def request_loop(root,profile,name,conversation_id=None):
    """A chat is one conversation: each line is a message, each answer names the agent answering next.

    stdin EOF only ends this loop; tasks already sent keep running, and the
    conversation can be reopened with its conversation_id.
    """
    from ..client import Client
    async with Client.connect(root,profile) as client:
        if conversation_id is not None:
            # Reattach: wait for whoever answers this conversation now.
            existing = await client.store.get_conversation(conversation_id)
            if existing is None:
                from .._infrastructure.processes import InfrastructureError
                raise InfrastructureError('conversation_not_found','no such conversation')
            name = existing.active_agent
        await wait_for_agent(client,name)
        conversation = await client.conversation(name,conversation_id)
        print(json.dumps({'conversation_id':conversation.conversation_id,
                          'active_agent':await conversation.refresh()}),flush=True)
        while True:
            try: text = await asyncio.to_thread(input,'')
            except EOFError: break
            if not text.strip(): continue
            try: value = json.loads(text)
            except ValueError:
                print(json.dumps({'error_code':'invalid_json','message':'each line must be one JSON value'}),flush=True)
                continue
            result = await conversation.ask(value)
            print(json.dumps(result.model_dump(mode='json')|{'active_agent':conversation.active_agent},
                             ensure_ascii=False),flush=True)
