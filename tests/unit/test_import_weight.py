"""A clean interpreter rejects network/process/config effects during core import."""
import subprocess
import sys
import os
from pathlib import Path


def test_core_import_has_no_io_or_optional_sdk():
    script = r'''
import sys, socket, subprocess, os
from pathlib import Path
blocked = ('openai', 'anthropic', 'google.genai', 'mcp', 'psycopg', 'confluent_kafka')
def deny(*a, **kw):
    raise AssertionError('core import attempted runtime I/O')
def audit(event, args):
    if event.startswith('socket.') or event in ('os.system', 'os.fork', 'os.posix_spawn', 'subprocess.Popen'):
        deny()
    if event == 'open':
        path, mode, flags = args
        if (mode and any(x in mode for x in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            deny()
        if isinstance(path, (str, bytes)):
            path = Path(os.fsdecode(path)).resolve()
            roots = (Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(), Path(os.environ['PYTHONPATH']).resolve() / 'agentweave')
            source = path.suffix in ('.py', '.pyc', '.so') and any(path.is_relative_to(root) for root in roots)
            metadata = path.parent.name.endswith('.dist-info') and path.name == 'entry_points.txt' and path.is_relative_to(Path(sys.prefix).resolve())
            if not (source or metadata):
                deny()
sys.addaudithook(audit)
socket.socket.connect = deny
socket.socket.bind = deny
subprocess.Popen = deny
original_getenv = os.getenv
def checked_getenv(name, *args):
    if name != 'PYDANTIC_DISABLE_PLUGINS':
        return deny()
    return original_getenv(name, *args)
os.getenv = checked_getenv
original_read_text = Path.read_text
def checked_read_text(path, *args, **kwargs):
    if path.parent.name.endswith('.dist-info') and path.name == 'entry_points.txt':
        return original_read_text(path, *args, **kwargs)
    return deny()
Path.read_text = checked_read_text
Path.read_bytes = deny
Path.write_bytes = deny
Path.write_text = deny
Path.mkdir = deny
import agentweave
assert {'Agent','Model','Tools','Peers','tool'} <= set(agentweave.__all__)
assert not any(any(name == prefix or name.startswith(prefix+'.') for prefix in blocked) for name in sys.modules)
a = agentweave.Agent('assistant', model=agentweave.Model.openai('chosen'))
assert a.name == 'assistant'
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=os.environ | {'PYTHONPATH':str(Path(__file__).resolve().parents[2] / 'src')})
    assert result.returncode == 0, result.stderr
