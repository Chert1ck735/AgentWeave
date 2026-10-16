"""The local broker is shipped in the wheel, never downloaded at runtime."""
from .locator import resolve_broker_bin

__all__ = ['resolve_broker_bin']
