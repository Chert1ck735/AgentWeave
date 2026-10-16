"""Stable text/JSON CLI boundary, with no shell-command interpolation."""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from pydantic import ValidationError
from ..errors import BrokerUnavailable, EnvelopeTooLarge, InvalidEnvelope, StoreUnavailable
from .._infrastructure.processes import InfrastructureError

# Stable exit codes: 0 success, 2 configuration or input, 3 infrastructure
# unavailable, 4 task failed or its outcome is unknown, 1 any other failure.
EXIT_INPUT, EXIT_UNAVAILABLE, EXIT_TASK, EXIT_OTHER = 2, 3, 4, 1
_INPUT_ERRORS = frozenset(('usage_error', 'invalid_input', 'invalid_json', 'missing_value', 'definition_invalid',
    'definition_exists', 'capability_mismatch', 'model_key_missing', 'model_unavailable', 'agent_not_found',
    'tool_not_found', 'conversation_not_found', 'profile_invalid', 'already_running', 'action_refused',
    'version_conflict', 'task_not_found'))
_UNAVAILABLE_ERRORS = frozenset(('infrastructure_unavailable', 'infrastructure_unmanaged', 'profile_unavailable',
    'broker_unavailable', 'broker_start_failed', 'broker_start_timeout', 'dev_extra_missing',
    'dev_binary_unavailable', 'ownership_mismatch', 'worker_start_failed', 'worker_start_timeout',
    'worker_exited', 'agent_unavailable', 'store_unavailable', 'journal_unavailable'))
_TASK_ERRORS = frozenset(('task_failed', 'task_unknown'))


def exit_code(error_code):
    if error_code in _INPUT_ERRORS: return EXIT_INPUT
    if error_code in _UNAVAILABLE_ERRORS: return EXIT_UNAVAILABLE
    if error_code in _TASK_ERRORS: return EXIT_TASK
    return EXIT_OTHER


def build_parser(json_mode=False):
    """The whole command tree; parsing only, nothing runs."""
    class Parser(argparse.ArgumentParser):
        def error(self,message):
            if json_mode:
                print(json.dumps({'error_code':'usage_error','message':'invalid command or arguments'}),flush=True)
                raise SystemExit(EXIT_INPUT)
            super().error(message)
    parser = Parser(prog='agentweave')
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--profile',choices=('dev','server'),default='dev')
    parser.add_argument('--json',action='store_true')
    commands = parser.add_subparsers(dest='group',required=True)
    dev = commands.add_parser('dev').add_subparsers(dest='action',required=True)
    run = dev.add_parser('run'); _launch_options(run)
    agents = commands.add_parser('agent').add_subparsers(dest='action',required=True)
    for name in ('run','start'):
        command = agents.add_parser(name); _launch_options(command)
    for name in ('stop','drain','restart','status','logs'):
        command = agents.add_parser(name); command.add_argument('name'); command.add_argument('--replica',dest='instance')
        if name=='restart': _override_options(command)
    agents.add_parser('list')
    chat = agents.add_parser('chat'); chat.add_argument('name'); chat.add_argument('--conversation')
    _create_options(agents.add_parser('create'))
    validate = agents.add_parser('validate'); validate.add_argument('target'); _override_options(validate)
    request = agents.add_parser('request'); request.add_argument('name'); _payload_options(request)
    request.add_argument('--no-wait',action='store_true',help='print the task_id once accepted, do not wait')
    events = commands.add_parser('event').add_subparsers(dest='action',required=True)
    publish = events.add_parser('publish'); publish.add_argument('topic'); _payload_options(publish)
    publish.add_argument('--key',help='ordering key; with policy latest, also the coalescing key')
    publish.add_argument('--event-time',help='ISO 8601 time the fact happened at its source')
    publish.add_argument('--expires-at',help='ISO 8601 time after which no agent admits the event')
    publish.add_argument('--source-version',type=int,help='monotonic version of the source snapshot')
    publish.add_argument('--wait',metavar='AGENT',help='also wait for this agent to finish the event')
    publish.add_argument('--timeout',type=float,default=60.0,help='seconds to wait with --wait')
    subscribe = events.add_parser('subscribe'); subscribe.add_argument('topic')
    subscribe.add_argument('--limit',type=int,help='stop after this many events')
    subscribe.add_argument('--from-retained',action='store_true',help='also print events the broker still retains')
    _task_commands(commands.add_parser('task').add_subparsers(dest='action',required=True))
    states = commands.add_parser('state').add_subparsers(dest='action',required=True)
    states.add_parser('inspect').add_argument('session_id')
    states.add_parser('export').add_argument('path',help='new SQLite file; an existing file is never overwritten')
    states.add_parser('cleanup').add_argument('--apply',action='store_true',help='delete; without it a dry run only reports')
    tools = commands.add_parser('tool').add_subparsers(dest='action',required=True)
    tools.add_parser('list')
    tools.add_parser('describe').add_argument('name')
    stop_tool = tools.add_parser('stop'); stop_tool.add_argument('name'); stop_tool.add_argument('--replica',dest='instance')
    infrastructure = commands.add_parser('infra').add_subparsers(dest='action',required=True)
    for name in ('up','down','status'): infrastructure.add_parser(name)
    commands.add_parser('doctor')
    child = commands.add_parser('_worker'); child.add_argument('target'); child.add_argument('instance'); child.add_argument('--overrides',default='{}')
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser('--json' in argv).parse_args(argv)
    args.root = args.root.resolve()
    def emit(value):
        print(json.dumps(value,ensure_ascii=False) if args.json else '\n'.join(f'{key}: {item}' for key,item in value.items()),flush=True)
    try:
        if args.group in ('task','state'):
            from importlib import import_module
            return import_module('.'+args.group,__package__).command(args.root,args.profile,args,emit)
        if args.group=='event':
            from . import event
            return getattr(event,args.action)(args.root,args.profile,args,emit)
        if args.group=='agent' and args.action=='create':
            from .definitions import create
            emit(create(args.root,args))
            return 0
        if args.group=='agent' and args.action=='validate':
            from .definitions import validate
            emit(validate(args.root,args.target,_overrides(args)))
            return 0
        if args.group=='agent' and args.action=='request':
            from .dev import request
            from .event import read_payload,session
            payload = read_payload(args.payload,args.file)
            return asyncio.run(request(args.root,args.profile,args.name,payload,session_id=session(args.session_id),
                                       wait=not args.no_wait,emit=emit))
        if args.group=='_worker':
            from .agent import worker_main
            asyncio.run(worker_main(args.target,args.instance,args.root,args.profile,json.loads(args.overrides)))
        elif args.group=='doctor':
            from .infra import doctor
            emit(doctor(args.root,args.profile))
        elif args.group=='infra':
            from . import infra
            emit(getattr(infra,args.action)(args.root,args.profile))
        elif args.group=='tool':
            from . import tool
            if args.action=='list': emit(tool.list_tools(args.root,args.profile))
            elif args.action=='describe': emit(tool.describe(args.root,args.profile,args.name))
            else: emit(tool.stop(args.root,args.profile,args.name,args.instance))
        elif args.group=='agent' and args.action=='chat':
            from .dev import request_loop
            asyncio.run(request_loop(args.root,args.profile,args.name,args.conversation))
        elif args.group=='dev' or args.action in ('run','start'):
            from .dev import run
            run(args.target,args.root,args.profile,detach=args.detach or args.action=='start',
                chat=args.chat,replica=args.replica,overrides=_overrides(args),emit=emit,
                stage=lambda value: None if args.json else print(value,file=sys.stderr,flush=True))
        else:
            from .agent import is_tool_record,select_records,stop_workers,start_worker
            agent_only = lambda record: not is_tool_record(record)  # tool services are managed by 'tool'
            registry,owned = select_records(args.root,args.profile,getattr(args,'name',None),getattr(args,'instance',None))
            records = [r for r in owned if agent_only(r)]
            if args.action in ('stop','drain'):
                emit(stop_workers(args.root,args.profile,args.name,args.instance,drain=args.action=='drain',only=agent_only))
            elif args.action=='restart':
                if not records: raise InfrastructureError('agent_not_found','agent is not registered')
                from ..config import resolve
                from .agent import load_definition
                launches=[(r.target,dict(r.overrides)|_overrides(args)) for r in records]
                for target,overrides in launches: load_definition(target,args.root,overrides)
                from .._infrastructure.bootstrap import Bootstrap,Paths
                config=resolve(args.root,args.profile)
                if ((Paths(args.root,args.profile).profile_dir/'managed.json').exists() or
                    any(r.logical_id=='$broker' for r in registry.records())):
                    config=Bootstrap(args.root,args.profile).prepare(config)
                stop_workers(args.root,args.profile,args.name,args.instance,only=agent_only)
                started=[asdict(start_worker(target,config,True,replica=True,overrides=overrides))
                         for target,overrides in launches]
                emit({'started':started})
            elif args.action=='logs':
                from .._infrastructure.bootstrap import Paths
                emit({'logs':[{ 'instance_id':r.instance_id,'text':(Paths(args.root,args.profile).logs/(r.instance_id+'.log')).read_text(errors='replace')[-65536:]} for r in owned]})
            else:
                emit({'agents':[asdict(r)|{'owned':registry.verify(r),'running':registry.running(r)} for r in records]})
        return 0
    except InfrastructureError as exc:
        emit({'error_code':exc.code,'message':str(exc)}|({'details':exc.details} if exc.details else {}))
        return exit_code(exc.code)
    except KeyboardInterrupt:
        emit({'error_code':'interrupted','message':'operation interrupted'})
        return 130
    except BrokenPipeError:
        # The reader went away (| head): stop quietly, without a second failing write.
        os.dup2(os.open(os.devnull,os.O_WRONLY),sys.stdout.fileno())
        return EXIT_OTHER
    except (BrokerUnavailable,StoreUnavailable) as exc:
        code = 'broker_unavailable' if isinstance(exc,BrokerUnavailable) else 'store_unavailable'
        emit({'error_code':code,'message':'the broker or the journal is unavailable'})
        return EXIT_UNAVAILABLE
    except (InvalidEnvelope,ValidationError) as exc:
        message = ('the input exceeds the envelope size limit' if isinstance(exc,EnvelopeTooLarge)
                   else 'the input does not form a valid event or request')
        emit({'error_code':'invalid_input','message':message})
        return EXIT_INPUT
    except Exception:
        emit({'error_code':'operation_failed','message':'operation failed; inspect private project logs'})
        return EXIT_OTHER


def _launch_options(parser):
    parser.add_argument('target')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--chat',action='store_true')
    parser.add_argument('--replica',action='store_true')
    parser.add_argument('--namespace',help='tool services: namespace of the published tool IDs')
    _override_options(parser)


def _task_commands(tasks):
    for name in ('status','result','events'):
        tasks.add_parser(name).add_argument('task_id')
    for name in ('cancel','fail','resolve','retry'):
        command = tasks.add_parser(name)
        command.add_argument('task_id')
        command.add_argument('--expected-version',type=int,required=True,help='the version task status showed')
        if name in ('cancel','fail'):
            command.add_argument('--reason',required=True)
            continue
        command.add_argument('--step',required=True,help='tool_call_id of the unknown call, or handler')
        command.add_argument('--deadline',help='ISO 8601 new deadline; required once the old one passed')
        if name=='resolve':
            command.add_argument('--result',help='the confirmed result as JSON')
            command.add_argument('--result-file',help='read the confirmed result from this JSON file')
            command.add_argument('--state',help='a handler step: its full confirmed session state as a JSON object')
        else:
            command.add_argument('--allow-unknown',action='store_true',help='accept that the unknown effect may repeat')


def _payload_options(parser):
    parser.add_argument('payload',nargs='?',help='JSON value; omit it to read --file or stdin')
    parser.add_argument('--file',help='read the JSON value from this file')
    parser.add_argument('--session-id',help='session UUID, for stateful agents')


def _create_options(parser):
    parser.add_argument('name')
    parser.add_argument('--non-interactive',action='store_true',help='never prompt; missing required values fail')
    parser.add_argument('--replace',action='store_true',help='overwrite an existing definition of NAME')
    parser.add_argument('--definition',metavar='FILE',help='JSON object of agent settings; - reads stdin')
    parser.add_argument('--instructions')
    parser.add_argument('--provider',choices=('openai','compatible','anthropic','gemini','custom'))
    parser.add_argument('--model',help='model id')
    parser.add_argument('--api-key-env',help='name of the environment variable holding the key, never the key')
    parser.add_argument('--base-url',help='endpoint of a compatible provider')
    parser.add_argument('--factory',help='custom adapter: registered name, entry point or module:object')
    parser.add_argument('--no-tool-calling',action='store_true',help='the model cannot call tools')
    parser.add_argument('--no-streaming',action='store_true',help='the endpoint cannot stream')
    parser.add_argument('--tool',action='append',default=[],help='allowed discovered tool or namespace.*')
    parser.add_argument('--all-tools',action='store_true',help='allow every discovered tool')
    parser.add_argument('--http-tool',action='append',default=[],help='HTTP tool specification as JSON')
    parser.add_argument('--peer',action='append',default=[],help='allowed peer agent')
    parser.add_argument('--discover-peers',action='store_true',help='allow every discovered peer')
    parser.add_argument('--max-tokens',type=int,help='token budget of a root task')
    parser.add_argument('--max-cost-usd',help='cost budget as a decimal string; needs pricing')
    parser.add_argument('--pricing-revision')
    parser.add_argument('--input-price',help='USD per million input tokens, decimal string')
    parser.add_argument('--output-price',help='USD per million output tokens, decimal string')
    _override_options(parser)


def _override_options(parser):
    parser.add_argument('--memory',choices=('stateless','stateful'))
    parser.add_argument('--concurrency',type=int)
    parser.add_argument('--max-inflight',type=int)


def _overrides(args):
    return {name:getattr(args,name) for name in ('memory','concurrency','max_inflight','namespace')
            if getattr(args,name,None) is not None}
