"""Shared fixtures. Tests run offline: outbound socket connections are blocked."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: Any) -> None:
    host = address[0] if isinstance(address, tuple) else address
    if host not in _LOOPBACK:  # asyncio's self-pipe uses loopback on Windows
        raise RuntimeError(f"network access blocked in tests: {address!r}")
    _real_connect(self, address)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    yield
