"""Serialization/framing utilities shared by SO-101 server/client scripts.

The DreamDifferent/OpenPI references use msgpack-numpy over websockets. The
mimic-video Euler env may not include ``msgpack`` or ``websockets``, so Phase 2
uses a dependency-free TCP transport by default and keeps msgpack as the
preferred payload format when available. If msgpack is unavailable, pickle is
used only for paired local server/dry-run clients on trusted connections.
"""

from __future__ import annotations

import functools
import pickle
import socket
import struct
from typing import Any

import numpy as np

try:
    import msgpack  # type: ignore
except ImportError:  # pragma: no cover - depends on the active conda env
    msgpack = None

PAYLOAD_PROTOCOL = "msgpack-numpy" if msgpack is not None else "pickle-numpy"
TCP_TRANSPORT = "tcp-length-prefixed"
_HEADER_STRUCT = struct.Struct("!Q")


def _pack_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported ndarray dtype for serialization: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported numpy scalar dtype for serialization: {obj.dtype}")
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_numpy(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


if msgpack is not None:
    packb = functools.partial(msgpack.packb, default=_pack_numpy, use_bin_type=True)
    unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_numpy, raw=False)
else:

    def packb(obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def unpackb(data: bytes) -> Any:
        return pickle.loads(data)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Socket closed while receiving framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_framed(sock: socket.socket, obj: Any) -> None:
    payload = packb(obj)
    sock.sendall(_HEADER_STRUCT.pack(len(payload)) + payload)


def recv_framed(sock: socket.socket) -> Any:
    header = recv_exact(sock, _HEADER_STRUCT.size)
    (payload_size,) = _HEADER_STRUCT.unpack(header)
    if payload_size <= 0:
        raise ValueError(f"Invalid framed payload size: {payload_size}")
    return unpackb(recv_exact(sock, payload_size))
