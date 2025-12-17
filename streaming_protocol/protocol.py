from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

MAGIC = b"SP01"
VERSION = 1

# Binary frame format (all integers are big-endian):
#   0  : 4 bytes  magic = b"SP01"
#   4  : 1 byte   version = 1
#   5  : 1 byte   flags (unused, 0)
#   6  : 2 bytes  reserved (0)
#   8  : 8 bytes  frame_id (uint64)
#  16  : 8 bytes  send_timestamp_ns (uint64)  # server monotonic time or wall time in ns; caller-defined
#  24  : 4 bytes  meta_len (uint32)           # UTF-8 JSON bytes
#  28  : 4 bytes  img1_len (uint32)
#  32  : 4 bytes  img2_len (uint32)
#  36  : 4 bytes  img3_len (uint32)
#  40  : meta bytes
#  ... : img1 bytes (JPEG)
#  ... : img2 bytes (JPEG)
#  ... : img3 bytes (JPEG)
#
# meta JSON schema (pydantic on both sides):
#   {"state": [float x 50], "timestamp_ns": int}
#
HEADER_STRUCT = struct.Struct(">4sBBHQQIIII")


@dataclass(frozen=True)
class FrameMessage:
    frame_id: int
    send_timestamp_ns: int
    meta: dict[str, Any]
    jpeg1: bytes
    jpeg2: bytes
    jpeg3: bytes


def encode_frame(
    *,
    frame_id: int,
    send_timestamp_ns: int,
    meta: dict[str, Any],
    jpeg1: bytes,
    jpeg2: bytes,
    jpeg3: bytes,
) -> bytes:
    meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        0,
        0,
        int(frame_id),
        int(send_timestamp_ns),
        len(meta_bytes),
        len(jpeg1),
        len(jpeg2),
        len(jpeg3),
    )
    return header + meta_bytes + jpeg1 + jpeg2 + jpeg3


def decode_frame(payload: bytes) -> FrameMessage:
    if len(payload) < HEADER_STRUCT.size:
        raise ValueError("payload too small for header")

    (
        magic,
        version,
        _flags,
        _reserved,
        frame_id,
        send_timestamp_ns,
        meta_len,
        img1_len,
        img2_len,
        img3_len,
    ) = HEADER_STRUCT.unpack_from(payload, 0)

    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}")

    total_len = HEADER_STRUCT.size + meta_len + img1_len + img2_len + img3_len
    if len(payload) != total_len:
        raise ValueError(f"payload length mismatch: got {len(payload)}, expected {total_len}")

    offset = HEADER_STRUCT.size
    meta_bytes = payload[offset : offset + meta_len]
    offset += meta_len
    jpeg1 = payload[offset : offset + img1_len]
    offset += img1_len
    jpeg2 = payload[offset : offset + img2_len]
    offset += img2_len
    jpeg3 = payload[offset : offset + img3_len]

    meta = json.loads(meta_bytes.decode("utf-8"))
    return FrameMessage(
        frame_id=int(frame_id),
        send_timestamp_ns=int(send_timestamp_ns),
        meta=meta,
        jpeg1=jpeg1,
        jpeg2=jpeg2,
        jpeg3=jpeg3,
    )
