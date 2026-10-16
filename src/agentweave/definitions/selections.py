"""Explicit access selections; absence and empty allow lists remain distinct."""
from typing import Literal
from pydantic import field_validator
from ..messages import StrictModel, validate_name

class ToolSelection(StrictModel):
    mode: Literal['discover'] = 'discover'
    allow: tuple[str, ...] | None = None

    @field_validator('allow', mode='before')
    @classmethod
    def names(cls, value):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise ValueError('allow must be a list of logical names or namespace.* patterns')
        return tuple(validate_name(x[:-2]) + '.*' if isinstance(x, str) and x.endswith('.*') else validate_name(x) for x in value)

class PeerSelection(StrictModel):
    mode: Literal['discover', 'allow']
    names: tuple[str, ...] = ()

    @field_validator('names', mode='before')
    @classmethod
    def validate_names(cls, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError('names must be a list')
        return tuple(validate_name(x) for x in value)

class Tools:
    @staticmethod
    def discover(allow=None) -> ToolSelection:
        return ToolSelection(allow=allow)

class Peers:
    @staticmethod
    def discover() -> PeerSelection:
        return PeerSelection(mode='discover')

    @staticmethod
    def allow(names) -> PeerSelection:
        return PeerSelection(mode='allow', names=names)
