"""Owned dev bootstrap and private project/profile paths."""
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4
from .processes import InfrastructureError, ProcessRegistry, atomic_json, file_lock, private_dir


@dataclass(frozen=True)
class Paths:
    root: Path
    profile: str = 'dev'

    @property
    def base(self): return self.root/'.agentweave'
    @property
    def profile_dir(self): return self.base/self.profile
    @property
    def registry(self): return self.profile_dir/'processes'
    @property
    def journal(self): return self.profile_dir/'state.sqlite3'
    @property
    def config(self): return self.profile_dir/'config.toml'
    @property
    def epochs(self): return self.profile_dir/'epochs'
    @property
    def logs(self): return self.profile_dir/'logs'

    def prepare(self):
        for path in (self.base,self.profile_dir,self.registry,self.logs,self.epochs):
            private_dir(path)


@dataclass
class RuntimeResources:
    config: object
    store: object
    bus: object
    directory: object = None

    def __post_init__(self):
        if self.directory is None:
            from .._directory import LiveDirectory
            self.directory = LiveDirectory(self.bus,self.store,self.config.network_id,
                broker_epoch=self.config.broker_epoch or '')

    async def close(self):
        try:
            await self.directory.close()
        finally:
            try:
                await self.bus.close()
            finally:
                await self.store.close()


class Bootstrap:
    def __init__(self,root=None,profile='dev'):
        self.root = Path(root or Path.cwd()).resolve()
        self.profile = profile

    async def ensure(self,config=None):
        config = await asyncio.to_thread(self.prepare,config)
        return await self.open_resources(config)

    @staticmethod
    async def open_resources(config):
        from ..errors import ConfigError
        if not config.database_url or not config.database_url.startswith('sqlite:///'):
            raise ConfigError('this runtime requires a configured SQLite backend')
        from ..adapters.storage.sqlite import SQLiteStore
        from ..adapters.broker import BrokerBus
        path = Path(config.database_url.removeprefix('sqlite:///'))
        if not path.is_file():
            raise ConfigError('configured database has not been prepared')
        store = SQLiteStore(path)
        store.retention = config.retention  # the replay window admission enforces
        bus = BrokerBus(config.broker_url,config.network_id,profile=config.profile,
                        broker_epoch=config.broker_epoch or '')
        resources = RuntimeResources(config,store,bus)
        try:
            await store.migrate()
            await resources.directory.start()
            return resources
        except BaseException:
            await resources.close()
            raise

    def prepare(self,config=None):
        from ..config import _read_project, resolve, write_config
        if self.profile != 'dev':
            raise InfrastructureError('profile_unavailable','the server profile is not supported in v0.1 (planned for v0.2); use --profile dev')
        paths = Paths(self.root,self.profile)
        from ..errors import ConfigError
        try:
            prepared = config or resolve(self.root,self.profile)
        except ConfigError:
            if paths.config.exists(): raise
            prepared = None
        owned = [record for record in ProcessRegistry(self.root,self.profile).records() if record.logical_id == '$broker']
        if prepared and prepared.broker_url and not owned and not (paths.profile_dir/'managed.json').exists():
            raise InfrastructureError('infrastructure_unmanaged','configured broker is not CLI-owned; connection authority does not authorize lifecycle control')
        paths.prepare()
        with file_lock(paths.base/'bootstrap.lock'):
            identity_path = paths.base/'project.json'
            project = _read_project(self.root)
            identity = project.project_id or (prepared.project_id if prepared is not None else None)
            if identity_path.exists():
                stored = json.loads(identity_path.read_text())['project_id']
                if identity is not None and identity != stored:
                    raise InfrastructureError('ownership_mismatch','project identity changed')
                identity = stored
            identity = identity or str(uuid4())
            atomic_json(identity_path,{'project_id':identity,'root':str(self.root)})
            with file_lock(paths.profile_dir/'bootstrap.lock'):
                if config is None:
                    overrides = {} if paths.config.exists() else {'project_id':identity,'network_id':str(uuid4())}
                    config = resolve(self.root,self.profile,overrides)
                if config.root != self.root or config.profile != self.profile or config.project_id != identity:
                    raise InfrastructureError('ownership_mismatch','profile configuration identity differs')
                # Persist identity before constructing verifiable child records.
                write_config(config,paths.config)
                # The managed journal is private project storage: never follow a link elsewhere.
                for candidate in (paths.journal, *(paths.journal.with_name(paths.journal.name + suffix)
                                                   for suffix in ('-wal', '-shm', '-journal'))):
                    if candidate.is_symlink():
                        raise InfrastructureError('ownership_mismatch','managed journal files must not be symlinks')
                from ..adapters.storage.sqlite import SQLiteStore
                async def migrate():
                    store = SQLiteStore(paths.journal)
                    try: await store.migrate()
                    finally: await store.close()
                asyncio.run(migrate())
                paths.journal.chmod(0o600)
                managed_path = paths.profile_dir/'managed.json'
                marker = {'root':str(self.root),'project_id':config.project_id,'network_id':config.network_id}
                if managed_path.exists() and json.loads(managed_path.read_text()) != marker:
                    raise InfrastructureError('ownership_mismatch','managed infrastructure identity differs')
                atomic_json(managed_path,marker)
                from .local import ensure_broker
                config = ensure_broker(config,paths)
                write_config(config,paths.config)
                async def provision():
                    from ..adapters.broker import BrokerBus
                    from ..ports import TopicSpec
                    bus = BrokerBus(config.broker_url,config.network_id,profile=config.profile,
                                    broker_epoch=config.broker_epoch or '')
                    try:
                        await bus.ensure_topics([TopicSpec(bus.control_topic(kind),partitions=1)
                            for kind in ('directory','receipt','progress','error')])
                    finally: await bus.close()
                asyncio.run(provision())
                return config
