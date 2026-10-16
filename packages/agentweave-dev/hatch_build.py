"""Make the binary wheel platform-specific and reject incomplete builds."""
import hashlib
import json
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        package = Path(self.root) / 'src/agentweave_dev'
        binary = package / '_bin/tansu'
        manifest = json.loads((package / 'manifest.json').read_text())
        if not binary.is_file():
            raise RuntimeError('Package the pinned official broker release before building its wheel')
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['sha256']:
            raise RuntimeError('Broker SHA256 differs from manifest')
        build_data['pure_python'] = False
        build_data['tag'] = 'py3-none-manylinux_2_39_x86_64'
