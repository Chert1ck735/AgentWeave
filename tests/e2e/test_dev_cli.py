"""Actual CLI children, broker processes, and permanent journal lifecycle."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from uuid import uuid4

import psutil
import pytest

REPO = Path(__file__).resolve().parents[2]
CREATED = {}
ENV = os.environ | {'PYTHONPATH':os.pathsep.join(map(str,[REPO/'src',REPO/'packages/agentweave-dev/src']))}


def live_pid(pid):
    try:
        return psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def cli(root, *args, timeout=65):
    result = subprocess.run([sys.executable,'-m','agentweave','--root',str(root),'--json',*args],
        cwd=REPO,env=ENV,capture_output=True,text=True,timeout=timeout)
    if result.returncode == 0:
        try:
            payload=json.loads(result.stdout)
            values=[payload] if 'pid' in payload else payload.get('started',[])
            from dataclasses import fields
            from agentweave._infrastructure.processes import ProcessRecord
            names={f.name for f in fields(ProcessRecord)}
            for value in values:
                if value.get('owned',value.get('ready',False)) or value in payload.get('started',[]):
                    record=ProcessRecord(**{k:v for k,v in value.items() if k in names})
                    CREATED.setdefault(Path(root).resolve(),{})[record.instance_id]=record
        except (ValueError,TypeError):
            pass
    return result


def ok(root,*args):
    result = cli(root,*args)
    assert result.returncode == 0, (result.stdout,result.stderr)
    return json.loads(result.stdout)


@pytest.fixture
def project(tmp_path):
    root = tmp_path/'project пробелы'
    root.mkdir()
    (root/'app.py').write_text('''from agentweave import Agent
counter = Agent('counter', memory='stateful')
@counter.on_request(retry='safe')
async def count(ctx, request):
    ctx.state['count'] = ctx.state.get('count', 0) + len(request.input)
    return ctx.state['count']
''')
    try:
        yield root
    finally:
        # Cleanup uses the production ownership proof, never arbitrary PIDs.
        if (root/'.agentweave/dev').exists():
            cli(root,'agent','stop','counter',timeout=70)
            cli(root,'infra','down',timeout=70)
        # Retain original creation identity independently of a mutable registry.
        from agentweave._infrastructure.processes import ProcessRegistry
        from agentweave._infrastructure.local import cleanup_epoch
        registry=ProcessRegistry(root)
        for record in CREATED.pop(root.resolve(),{}).values():
            if registry.running(record):
                assert registry.verify(record), 'unverifiable live test process: '+str(record.pid)
                registry.terminate(record,grace=2)
            if record.logical_id=='$broker' and Path(record.epoch_dir).exists():
                cleanup_epoch(record,registry.paths)
            registry.remove(record)


def test_fresh_detach_roundtrip_reuse_and_preserved_journal(project):
    first = ok(project,'dev','run','app:counter','--detach')
    assert first['logical_id'] == 'counter'
    status = ok(project,'infra','status')
    assert status['owned'] and status['running']
    epoch_dir = Path(status['epoch_dir'])
    assert epoch_dir.is_dir() and (epoch_dir/'broker.sqlite').is_file()
    assert (epoch_dir.stat().st_mode & 0o777) == 0o700
    assert (project/'.agentweave/dev/state.sqlite3').is_file()
    again = cli(project,'agent','start','app:counter')
    assert again.returncode != 0 and json.loads(again.stdout)['error_code'] == 'already_running'
    async def ask():
        from agentweave import Client
        async with Client.connect(project) as client:
            async with asyncio.timeout(20):
                while not client.directory.snapshot().get('counter'):
                    await asyncio.sleep(.05)
                result = await client.agent('counter').ask([1,2],str(uuid4()))
                assert result.output == 2
    asyncio.run(ask())
    ok(project,'agent','stop','counter')
    assert ok(project,'infra','status')['pid'] == status['pid']
    second = ok(project,'dev','run','app:counter','--detach')
    assert second['pid'] != first['pid']
    assert ok(project,'infra','status')['broker_epoch'] == status['broker_epoch']
    ok(project,'agent','stop','counter')
    ok(project,'infra','down')
    assert not epoch_dir.exists()
    assert (project/'.agentweave/dev/state.sqlite3').is_file()
    restarted = ok(project,'infra','up')
    assert restarted['broker_epoch'] != status['broker_epoch']


def test_concurrent_bootstrap_reuses_one_owned_broker(project):
    argv = [sys.executable,'-m','agentweave','--root',str(project),'--json','infra','up']
    children = [subprocess.Popen(argv,cwd=REPO,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    results = [child.communicate(timeout=45) for child in children]
    assert [p.returncode for p in children] == [0,0],results
    values = [json.loads(out) for out,err in results]
    assert values[0]['pid'] == values[1]['pid']
    assert values[0]['network_id'] == values[1]['network_id']


def test_stop_does_not_kill_reused_pid(project):
    from agentweave._infrastructure.processes import ProcessRegistry, ProcessRecord
    from agentweave.config import resolve
    ok(project,'infra','up')
    config = resolve(project)
    foreign = subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
    try:
        record = ProcessRecord(config.project_id,'dev','counter',str(uuid4()),foreign.pid,
            psutil.Process(foreign.pid).create_time()-100,'counterfeit')
        ProcessRegistry(project).save(record)
        # The recorded worker is gone and its PID belongs to an unrelated
        # process: the stale record is dropped without signalling anyone.
        result = ok(project,'agent','stop','counter')
        assert result['stopped'] == [record.instance_id]
        assert foreign.poll() is None
        assert not [r for r in ProcessRegistry(project).records() if r.logical_id == 'counter']
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_invalid_definition_precedes_infrastructure(project):
    result = cli(project,'dev','run','app:missing','--detach')
    assert result.returncode != 0
    assert json.loads(result.stdout)['error_code'] == 'definition_invalid'
    assert not (project/'.agentweave').exists()


def test_foreign_port_is_not_touched(project):
    ok(project,'infra','up')
    old = ok(project,'infra','status')
    ok(project,'infra','down')
    port = int(old['broker_url'].split(':')[1])
    with socket.socket() as foreign:
        foreign.bind(('127.0.0.1',port)); foreign.listen()
        fresh = ok(project,'infra','up')
        assert int(fresh['broker_url'].split(':')[1]) != port
        assert foreign.getsockname()[1] == port


def test_unmanaged_prepared_endpoint_is_connection_only(project,broker_process):
    from agentweave.config import ProfileConfig,write_config
    from agentweave.adapters.storage.sqlite import SQLiteStore
    database = project/'manual.sqlite3'
    async def migrate():
        store=SQLiteStore(database)
        try: await store.migrate()
        finally: await store.close()
    asyncio.run(migrate())
    config=ProfileConfig(root=project,profile='dev',project_id=str(uuid4()),network_id=str(uuid4()),
        broker_url=broker_process.url,broker_epoch=str(uuid4()),database_url='sqlite:///'+str(database))
    write_config(config,project/'.agentweave/dev/config.toml')
    original=(project/'.agentweave/dev/config.toml').read_bytes()
    result=cli(project,'infra','up')
    assert result.returncode != 0
    assert json.loads(result.stdout)['error_code']=='infrastructure_unmanaged'
    assert (project/'.agentweave/dev/config.toml').read_bytes()==original
    assert broker_process.process.poll() is None
    assert not (project/'.agentweave/dev/state.sqlite3').exists()


def test_foreground_signal_stops_only_worker_and_symlink_reuses_network(project,tmp_path):
    alias=tmp_path/'alias'
    alias.symlink_to(project,target_is_directory=True)
    argv=[sys.executable,'-m','agentweave','--root',str(alias),'--json','dev','run','app:counter']
    process=subprocess.Popen(argv,cwd=REPO,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        line=process.stdout.readline()
        record=json.loads(line)
        assert record['ready']
        broker=ok(project,'infra','status')
        process.send_signal(signal.SIGHUP)
        process.communicate(timeout=40)
        assert process.returncode==0
        assert not live_pid(record['pid'])
        assert ok(alias,'infra','up')['pid']==broker['pid']
        assert (project/'.agentweave/dev/state.sqlite3').is_file()
    finally:
        if process.poll() is None:
            process.terminate(); process.wait(timeout=40)


def test_ambient_env_never_redirects_native_storage(project,tmp_path):
    redirected=tmp_path/'foreign.sqlite'
    (project/'.env').write_text('STORAGE_ENGINE=sqlite:///'+str(redirected)+'\nLISTENER_URL=tcp://0.0.0.0:19092\n')
    ambient=tmp_path/'ambient'; ambient.mkdir()
    (ambient/'.env').write_text('STORAGE_ENGINE=sqlite:///'+str(redirected)+'\nLISTENER_URL=tcp://0.0.0.0:19093\n')
    result=subprocess.run([sys.executable,'-m','agentweave','--root',str(project),'--json','infra','up'],
        cwd=ambient,env=ENV,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,(result.stdout,result.stderr)
    value=json.loads(result.stdout)
    assert value['broker_url'].startswith('127.0.0.1:')
    assert (Path(value['epoch_dir'])/'broker.sqlite').is_file()
    assert not redirected.exists()
    process=psutil.Process(value['pid'])
    assert 'STORAGE_ENGINE' not in process.environ()
    assert process.cwd()==value['epoch_dir']


def test_missing_extra_reports_install_command_without_download(project):
    env=ENV|{'PYTHONPATH':str(REPO/'src')}
    result=subprocess.run([sys.executable,'-m','agentweave','--root',str(project),'--json','infra','up'],
        cwd=REPO,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==3
    value=json.loads(result.stdout)
    assert value['error_code']=='dev_extra_missing'
    assert 'pip install' in value['message'] and 'agentweave[dev]' in value['message']
    assert not list((project/'.agentweave/dev/epochs').glob('*'))


def test_owned_broker_metadata_cannot_be_downgraded_to_external(project):
    from agentweave._infrastructure.processes import ProcessRegistry
    ok(project,'infra','up')
    registry=ProcessRegistry(project)
    record=next(r for r in registry.records() if r.logical_id=='$broker')
    from dataclasses import replace
    registry.save(replace(record,executable_hash='counterfeit'))
    try:
        script='''import asyncio,sys
from agentweave import Client
async def main():
    try:
        async with Client.connect(sys.argv[1]): pass
    except Exception as exc:
        print(getattr(exc,'code',type(exc).__name__))
        return
    raise SystemExit('unsafe success')
asyncio.run(main())
'''
        result=subprocess.run([sys.executable,'-c',script,str(project)],env=ENV,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert result.stdout.strip()=='ownership_mismatch'
        assert psutil.Process(record.pid).is_running()
    finally: registry.save(record)


def test_stop_proves_project_even_when_pid_time_and_executable_match(project):
    from agentweave._infrastructure.processes import ProcessRegistry,ProcessRecord,executable_hash
    from agentweave.config import resolve
    ok(project,'infra','up')
    config=resolve(project)
    argv=[sys.executable,'-c','import time; time.sleep(120)']
    foreign=subprocess.Popen(argv,cwd=project)
    try:
        process=psutil.Process(foreign.pid)
        record=ProcessRecord(config.project_id,'dev','counter',str(uuid4()),foreign.pid,
            process.create_time(),executable_hash(process.exe()),root=str(project),cwd=str(project),
            argv=argv,network_id=config.network_id)
        ProcessRegistry(project).save(record)
        result=cli(project,'agent','stop','counter')
        assert result.returncode==3
        assert json.loads(result.stdout)['error_code']=='ownership_mismatch'
        assert foreign.poll() is None
    finally:
        foreign.terminate(); foreign.wait(timeout=5)


def test_private_data_symlink_is_rejected(project,tmp_path):
    outside=tmp_path/'outside'; outside.mkdir()
    (project/'.agentweave').symlink_to(outside,target_is_directory=True)
    result=cli(project,'infra','up')
    assert result.returncode==3
    assert json.loads(result.stdout)['error_code']=='ownership_mismatch'
    assert not list(outside.iterdir())


@pytest.mark.parametrize('kind',['async','native'])
def test_stop_bounds_uncooperative_owned_worker_without_cancelling_journal(project,kind):
    # Regression target: relying on Python coroutine/thread cancellation hangs the CLI.
    body = '''from agentweave import Agent
import asyncio, threading, time
from pathlib import Path
counter=Agent('counter')
'''
    if kind=='async':
        body+='''@counter.on_request()
async def stuck(ctx,request):
    Path('entered').touch()
    while not Path('release').exists():
        try: await asyncio.sleep(.05)
        except asyncio.CancelledError: continue
    return 'released'
'''
    else:
        body+='''@counter.on_request()
def stuck(ctx,request):
    event=threading.Event()
    def release():
        while not Path('release').exists(): time.sleep(.05)
        event.set()
    threading.Thread(target=release,daemon=True).start()
    Path('entered').touch()
    event.wait()
    return 'released'
'''
    (project/'app.py').write_text(body)
    record=ok(project,'dev','run','app:counter','--detach')
    task_id=None
    async def submit():
        from agentweave import Client
        async with Client.connect(project) as client:
            async with asyncio.timeout(20):
                while not client.directory.snapshot().get('counter'): await asyncio.sleep(.05)
                submission=await client.agent('counter').submit('block')
                while not (project/'entered').exists(): await asyncio.sleep(.05)
                return submission.task_id
    try:
        task_id=asyncio.run(submit())
        before=time.monotonic()
        result=ok(project,'agent','stop','counter')
        assert time.monotonic()-before<37
        assert record['instance_id'] in result['stopped']
        assert not live_pid(record['pid'])
        async def journal():
            from agentweave import Client
            async with Client.connect(project) as client:
                task=await client.store.get_task(task_id)
                assert task is not None
                assert task.status not in ('cancelled','expired','succeeded')
        asyncio.run(journal())
        assert ok(project,'infra','status')['running']
    finally:
        (project/'release').touch()


def test_agent_replica_status_drain_restart_and_logs(project):
    first=ok(project,'agent','start','app:counter')
    second=ok(project,'agent','start','app:counter','--replica')
    assert first['instance_id']!=second['instance_id']
    listed=ok(project,'agent','list')['agents']
    assert len(listed)==2 and all(r['owned'] for r in listed)
    ok(project,'agent','drain','counter','--replica',second['instance_id'])
    assert len(ok(project,'agent','status','counter')['agents'])==1
    restarted=ok(project,'agent','restart','counter','--replica',first['instance_id'])
    assert len(restarted['started'])==1
    assert restarted['started'][0]['instance_id']!=first['instance_id']
    assert len(ok(project,'agent','logs','counter')['logs'])==1
    assert ok(project,'doctor')['binary_available']


def test_model_preflight_key_presence_and_unsupported_execution_do_no_infra_io(project):
    # OpenAI is available in v0.1; a native Anthropic adapter is deferred to v0.2.
    (project/'app.py').write_text("from agentweave import Agent,Model\ncounter=Agent('counter',model=Model.anthropic('test',api_key_env='TASK7_MODEL_KEY'))\n")
    argv=[sys.executable,'-m','agentweave','--root',str(project),'--json','dev','run','app:counter','--detach']
    for value,want in [(None,'model_key_missing'),('not-a-real-key','model_unavailable')]:
        env=dict(ENV)
        if value: env['TASK7_MODEL_KEY']=value
        else: env.pop('TASK7_MODEL_KEY',None)
        result=subprocess.run(argv,cwd=REPO,env=env,capture_output=True,text=True,timeout=10)
        assert result.returncode==2
        assert json.loads(result.stdout)['error_code']==want
        assert 'not-a-real-key' not in result.stdout+result.stderr
        assert not (project/'.agentweave').exists()


def test_cli_json_parse_errors_have_stable_error_codes(project):
    result=cli(project,'agent','nonsense')
    assert result.returncode==2
    assert json.loads(result.stdout)['error_code']=='usage_error'


def test_infra_up_provisions_control_topics_before_success(project):
    from confluent_kafka.admin import AdminClient
    status=ok(project,'infra','up')
    topics=AdminClient({'bootstrap.servers':status['broker_url'],'log_level':0}).list_topics(timeout=5).topics
    prefix='aw.'+status['network_id']+'.'
    assert {prefix+x for x in ('directory','receipts','progress','errors')} <= set(topics)


def test_new_epoch_loses_unadmitted_events_and_keeps_admitted_tasks(project):
    from datetime import datetime,timedelta,timezone
    from agentweave.messages import Envelope
    from agentweave import Client
    value=ok(project,'infra','up')
    fingerprint=subprocess.run([sys.executable,'-c',
        "from pathlib import Path; from agentweave.cli.agent import load_definition; from agentweave.definitions import fingerprint; import sys; print(fingerprint(load_definition('app:counter',Path(sys.argv[1]))))",str(project)],
        env=ENV,capture_output=True,text=True,timeout=10)
    assert fingerprint.returncode==0,fingerprint.stderr
    now=datetime.now(timezone.utc)
    event=Envelope(event_id=str(uuid4()),network_id=value['network_id'],kind='numbers.received',
        sender='client.test',recipient=None,topic='numbers.received',task_id=None,parent_task_id=None,
        correlation_id=None,session_id=str(uuid4()),created_at=now,deadline=now+timedelta(seconds=120),
        traceparent=None,payload={'numbers':[1]})
    accepted=Envelope(event_id=str(uuid4()),network_id=value['network_id'],kind='request',sender='client.test',
        recipient='counter',topic=None,task_id=str(uuid4()),parent_task_id=None,correlation_id=None,
        session_id=str(uuid4()),created_at=now,deadline=now+timedelta(seconds=120),traceparent=None,payload=[1,2,3])
    async def publish_and_admit():
        async with Client.connect(project) as client:
            await client.bus.publish(event)
            await client.store.admit_with_policy(accepted,fingerprint.stdout.strip(),agent_id='counter',max_inflight=64)
    asyncio.run(publish_and_admit())
    ok(project,'infra','down'); fresh=ok(project,'infra','up')
    assert fresh['broker_epoch']!=value['broker_epoch']
    override_conflict=cli(project,'agent','start','app:counter','--memory','stateless')
    assert override_conflict.returncode==3
    assert json.loads(override_conflict.stdout)['error_code']=='worker_start_failed'
    # Changed definition is rejected while accepted work exists, before readiness.
    original=(project/'app.py').read_text()
    (project/'app.py').write_text(original.replace("memory='stateful'","memory='stateless'"))
    conflict=cli(project,'agent','start','app:counter')
    assert conflict.returncode==3
    assert json.loads(conflict.stdout)['error_code']=='worker_start_failed'
    (project/'app.py').write_text(original)
    ok(project,'agent','start','app:counter')
    async def recovered():
        async with Client.connect(project) as client:
            async with asyncio.timeout(20):
                assert (await client.result(accepted.task_id)).output==3
            from uuid import UUID,uuid5
            lost_id=str(uuid5(UUID(event.event_id),'task:counter'))
            assert await client.store.get_task(lost_id) is None
    asyncio.run(recovered())


def test_infra_up_rejects_foreign_process_record_without_deleting_storage(project):
    from dataclasses import replace
    from agentweave._infrastructure.processes import ProcessRegistry
    value=ok(project,'infra','up')
    registry=ProcessRegistry(project)
    record=next(r for r in registry.records() if r.logical_id=='$broker')
    registry.save(replace(record,create_time=record.create_time-100))
    try:
        result=cli(project,'infra','up')
        assert result.returncode==3
        assert json.loads(result.stdout)['error_code']=='ownership_mismatch'
        assert (Path(value['epoch_dir'])/'broker.sqlite').exists()
        assert psutil.Process(record.pid).is_running()
    finally: registry.save(record)


def test_nested_python_target_uses_existing_definition_resolver(project):
    with (project/'app.py').open('a') as stream:
        stream.write('\nclass Catalog:\n    counter=counter\n')
    value=ok(project,'agent','start','app:Catalog.counter')
    assert value['logical_id']=='counter' and value['ready']


def test_drain_reports_unresolved_work_and_preserves_the_task(project):
    from datetime import datetime,timedelta,timezone
    from agentweave import Client
    from agentweave.messages import Envelope
    from agentweave._records import StepCommit
    ok(project,'infra','up')
    from agentweave.cli.agent import load_definition
    from agentweave.definitions import fingerprint
    definition_hash=fingerprint(load_definition('app:counter',project))
    task_id=str(uuid4())
    async def unresolved():
        async with Client.connect(project) as client:
            now=datetime.now(timezone.utc)
            event=Envelope(event_id=str(uuid4()),network_id=client.config.network_id,kind='request',
                sender='client.test',recipient='counter',topic=None,task_id=task_id,parent_task_id=None,
                correlation_id=None,session_id=str(uuid4()),created_at=now,deadline=now+timedelta(seconds=120),
                traceparent=None,payload=[])
            await client.store.admit_with_policy(event,definition_hash,agent_id='counter',max_inflight=64)
            lease=await client.store.claim(task_id,str(uuid4()),30)
            task=await client.store.get_task(task_id)
            await client.store.commit_step(lease,StepCommit(step_id='unknown',expected_version=task.version,status='outcome_unknown'))
    asyncio.run(unresolved())
    ok(project,'agent','start','app:counter')
    result=cli(project,'agent','drain','counter')
    assert result.returncode==1
    assert json.loads(result.stdout)['error_code']=='drain_incomplete'
    async def verify():
        async with Client.connect(project) as client:
            assert (await client.store.get_task(task_id)).status=='outcome_unknown'
    asyncio.run(verify())


def test_child_definition_failure_does_not_leave_adapter_threads_alive(project):
    with (project/'app.py').open('a') as stream:
        stream.write("\nimport os\nif os.environ.get('AGENTWEAVE_INSTANCE_ID'):\n    counter=object()\n")
    result=cli(project,'agent','start','app:counter',timeout=8)
    assert result.returncode==3
    assert json.loads(result.stdout)['error_code']=='worker_start_failed'


def test_cli_overrides_match_worker_and_survive_restart(project):
    started=ok(project,'dev','run','app:counter','--detach','--memory','stateless','--concurrency','2','--max-inflight','3')
    assert started['overrides']=={'memory':'stateless','concurrency':2,'max_inflight':3}
    async def counts():
        from agentweave import Client
        session=str(uuid4())
        async with Client.connect(project) as client:
            async with asyncio.timeout(20):
                while not client.directory.snapshot().get('counter'): await asyncio.sleep(.05)
                return [(await client.agent('counter').ask([1],session)).output for _ in range(2)]
    assert asyncio.run(counts())==[1,1]
    restarted=ok(project,'agent','restart','counter')['started'][0]
    assert restarted['overrides']==started['overrides']
    assert asyncio.run(counts())==[1,1]
    updated=ok(project,'agent','restart','counter','--memory','stateful')['started'][0]
    assert updated['overrides']=={'memory':'stateful','concurrency':2,'max_inflight':3}
    assert asyncio.run(counts())==[1,2]
    before=ok(project,'agent','status','counter')['agents'][0]['pid']
    invalid=cli(project,'agent','restart','counter','--concurrency','0')
    assert invalid.returncode==2
    assert json.loads(invalid.stdout)['error_code']=='definition_invalid'
    assert ok(project,'agent','status','counter')['agents'][0]['pid']==before


@pytest.mark.parametrize('flag,value',[('--memory','bad'),('--concurrency','0'),('--max-inflight','0')])
def test_invalid_cli_overrides_are_rejected_before_bootstrap(project,flag,value):
    result=cli(project,'dev','run','app:counter','--detach',flag,value)
    assert result.returncode in (1,2)
    assert json.loads(result.stdout)['error_code'] in ('usage_error','definition_invalid')
    assert not (project/'.agentweave').exists()


def test_restart_reconciles_dead_owned_broker(project):
    from agentweave._infrastructure.processes import ProcessRegistry
    ok(project,'agent','start','app:counter')
    registry=ProcessRegistry(project)
    broker=next(r for r in registry.records() if r.logical_id=='$broker')
    registry.terminate(broker,grace=2)
    value=ok(project,'agent','restart','counter')
    assert value['started'][0]['broker_epoch']!=broker.broker_epoch
    assert ok(project,'infra','status')['owned']


def test_malformed_epoch_owner_cannot_delete_permanent_journal(project):
    from agentweave._infrastructure.bootstrap import Paths
    from agentweave._infrastructure.local import cleanup_epoch
    from agentweave._infrastructure.processes import InfrastructureError,ProcessRecord
    paths=Paths(project); paths.prepare()
    paths.journal.write_bytes(b'permanent journal sentinel')
    instance=str(uuid4()); identity=str(uuid4())
    record=ProcessRecord(identity,'dev','$broker',instance,99999999,0,'not-running',root=str(project),
        broker_epoch='..',epoch_dir=str(paths.epochs/'..'))
    (paths.profile_dir/'owner.json').write_text(json.dumps({'root':str(project),'project_id':identity,
        'instance_id':instance,'broker_epoch':'..'}))
    with pytest.raises(InfrastructureError,match='epoch'):
        cleanup_epoch(record,paths)
    assert paths.journal.read_bytes()==b'permanent journal sentinel'


def test_symlink_epoch_parent_cannot_delete_other_directory(project,tmp_path):
    from agentweave._infrastructure.bootstrap import Paths
    from agentweave._infrastructure.local import cleanup_epoch
    from agentweave._infrastructure.processes import InfrastructureError,ProcessRecord
    paths=Paths(project); paths.prepare()
    foreign=tmp_path/'unrelated'; foreign.mkdir()
    epoch=str(uuid4()); instance=str(uuid4()); identity=str(uuid4())
    target=foreign/epoch; target.mkdir()
    sentinel=target/'sentinel'; sentinel.write_bytes(b'unrelated data')
    (target/'owner.json').write_text(json.dumps({'root':str(project),'project_id':identity,
        'instance_id':instance,'broker_epoch':epoch}))
    paths.epochs.rmdir(); paths.epochs.symlink_to(foreign,target_is_directory=True)
    record=ProcessRecord(identity,'dev','$broker',instance,99999999,0,'not-running',root=str(project),
        broker_epoch=epoch,epoch_dir=str(paths.epochs/epoch))
    with pytest.raises(InfrastructureError,match='epoch'):
        cleanup_epoch(record,paths)
    assert sentinel.read_bytes()==b'unrelated data'


def test_broker_start_failure_after_spawn_leaves_no_process(project,monkeypatch):
    from agentweave._infrastructure import local
    from agentweave._infrastructure.bootstrap import Bootstrap
    from agentweave._infrastructure.processes import InfrastructureError
    spawned=[]
    real_popen=local.subprocess.Popen
    def popen(argv,*args,**kwargs):
        child=real_popen(argv,*args,**kwargs)
        if '--listener-url' in argv: spawned.append(child)  # the broker, not --version
        return child
    def failing_capture(*args,**kwargs):
        raise InfrastructureError('capture_failed','process identity could not be recorded')
    monkeypatch.setattr(local.subprocess,'Popen',popen)
    monkeypatch.setattr(local,'capture',failing_capture)
    with pytest.raises(InfrastructureError,match='identity'):
        Bootstrap(project).prepare()
    assert spawned and spawned[0].poll() is not None, 'unrecorded broker child is still running'
    assert not any((project/'.agentweave/dev/epochs').iterdir())


def test_worker_start_failure_after_spawn_leaves_no_process(project,monkeypatch):
    from agentweave.cli import agent
    from agentweave._infrastructure.bootstrap import Bootstrap
    from agentweave._infrastructure.processes import InfrastructureError
    config=Bootstrap(project).prepare()
    spawned=[]
    real_popen=agent.subprocess.Popen
    def popen(argv,*args,**kwargs):
        spawned.append(real_popen(argv,*args,**kwargs))
        return spawned[-1]
    def failing_capture(*args,**kwargs):
        raise InfrastructureError('capture_failed','process identity could not be recorded')
    monkeypatch.setattr(agent.subprocess,'Popen',popen)
    monkeypatch.setattr(agent,'capture',failing_capture)
    with pytest.raises(InfrastructureError,match='identity'):
        agent.start_worker('app:counter',config,detach=True)
    assert spawned and spawned[0].poll() is not None, 'unrecorded worker child is still running'


def test_dead_broker_whose_pid_was_reused_is_reconciled(project):
    from dataclasses import replace
    from agentweave._infrastructure.processes import ProcessRegistry
    first=ok(project,'infra','up')
    registry=ProcessRegistry(project)
    broker=next(r for r in registry.records() if r.logical_id=='$broker')
    registry.terminate(broker,grace=2)
    foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
    try:
        # The dead broker's PID now belongs to an unrelated process.
        registry.save(replace(broker,pid=foreign.pid))
        second=ok(project,'infra','up')
        assert second['broker_epoch']!=first['broker_epoch']
        assert not Path(first['epoch_dir']).exists()
        assert foreign.poll() is None
    finally:
        foreign.terminate(); foreign.wait(timeout=5)


def test_live_broker_missing_from_config_is_adopted(project):
    from agentweave._infrastructure.bootstrap import Paths
    from agentweave.config import resolve,write_config
    first=ok(project,'infra','up')
    config=resolve(project)
    # A first run killed after saving the broker record but before writing config.
    write_config(config.model_copy(update={'broker_url':None,'broker_epoch':None}),Paths(project).config)
    second=ok(project,'infra','up')
    assert second['broker_epoch']==first['broker_epoch'] and second['pid']==first['pid']
    assert resolve(project).broker_epoch==first['broker_epoch']


def test_foreground_run_reports_a_crashed_worker(project):
    from agentweave._infrastructure.processes import ProcessRegistry
    child=subprocess.Popen([sys.executable,'-m','agentweave','--root',str(project),'--json','dev','run','app:counter'],
        cwd=REPO,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        ready=json.loads(child.stdout.readline())
        record=next(r for r in ProcessRegistry(project).records() if r.instance_id==ready['instance_id'])
        CREATED.setdefault(project.resolve(),{})[record.instance_id]=record
        os.kill(record.pid,signal.SIGKILL)  # a crash, not a requested stop
        out,_=child.communicate(timeout=30)
        assert child.returncode!=0
        assert json.loads(out.strip().splitlines()[-1])['error_code']=='worker_exited'
    finally:
        if child.poll() is None: child.kill(); child.wait(timeout=5)


def test_broker_start_cleanup_failure_keeps_the_original_error(project,monkeypatch):
    from agentweave._infrastructure import local
    from agentweave._infrastructure.bootstrap import Bootstrap
    from agentweave._infrastructure.processes import InfrastructureError,ProcessRegistry
    def never_ready(url,timeout=1):
        raise InfrastructureError('probe_failed','readiness probe failed')
    def stuck(self,record,**kwargs):
        raise InfrastructureError('stop_timeout','owned process has not exited')
    monkeypatch.setattr(local,'ready',never_ready)
    monkeypatch.setattr(ProcessRegistry,'terminate',stuck)
    with pytest.raises(InfrastructureError) as raised:
        Bootstrap(project).prepare()
    assert raised.value.code=='probe_failed'
    assert not [r for r in ProcessRegistry(project).records() if r.logical_id=='$broker']
    assert not any((project/'.agentweave/dev/epochs').iterdir())


def test_supported_openai_agent_starts_through_the_cli(project,monkeypatch):
    (project/'assistant.py').write_text("from agentweave import Agent,Model\n"
        "assistant=Agent('assistant',model=Model.openai('test-model',api_key_env='AW_TEST_OPENAI_KEY'))\n")
    monkeypatch.setitem(ENV,'AW_TEST_OPENAI_KEY','sk-not-a-real-key')  # never used: no request is sent
    started=ok(project,'dev','run','assistant:assistant','--detach')
    assert started['logical_id']=='assistant' and started['ready']
    assert ok(project,'agent','stop','assistant')['stopped']==[started['instance_id']]


def test_sigterm_during_foreground_startup_stops_the_worker(project):
    from agentweave._infrastructure.processes import ProcessRegistry
    ok(project,'infra','up')  # keep broker startup out of the signal window
    child=subprocess.Popen([sys.executable,'-m','agentweave','--root',str(project),'--json','dev','run','app:counter'],
        cwd=REPO,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    registry=ProcessRegistry(project)
    deadline=time.monotonic()+20
    worker=None
    while worker is None and time.monotonic()<deadline:
        worker=next((r for r in registry.records() if r.logical_id=='counter'),None)
        time.sleep(.01)
    assert worker is not None
    CREATED.setdefault(project.resolve(),{})[worker.instance_id]=worker
    child.send_signal(signal.SIGTERM)  # before the worker reports ready
    child.communicate(timeout=60)
    assert child.returncode!=0
    assert not live_pid(worker.pid), 'foreground startup left its worker running'


def test_symlinked_journal_is_refused_before_migration(project,tmp_path):
    import sqlite3
    outside=tmp_path/'someone-else.sqlite3'
    with sqlite3.connect(outside) as connection: connection.execute('CREATE TABLE mine(x)')
    outside.chmod(0o644)
    profile=project/'.agentweave'/'dev'
    profile.mkdir(parents=True,mode=0o700); (project/'.agentweave').chmod(0o700)
    (profile/'state.sqlite3').symlink_to(outside)
    result=cli(project,'infra','up')
    assert result.returncode==3 and json.loads(result.stdout)['error_code']=='ownership_mismatch'
    with sqlite3.connect(outside) as connection:
        assert [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]==['mine']
    assert (outside.stat().st_mode & 0o777)==0o644
