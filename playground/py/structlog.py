"""
Stand-in for structlog in the gateway playground.

The gateway logs every verdict; in a browser tab there is nowhere for those
lines to go, and installing the real package would mean fetching it from PyPI
on every visit. This module offers the two calls the gateway makes,
get_logger() and bind(), and drops everything.
"""

from __future__ import annotations

from typing import Any


class _Discard:
    def bind(self, **_: Any) -> _Discard:
        return self

    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None


def get_logger(*_args: Any, **_kwargs: Any) -> _Discard:
    return _Discard()
