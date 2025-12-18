from __future__ import annotations

import json
import struct
from enum import IntEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ImageRole(IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    BACK = 3


MAGIC: bytes = b"SP02"
VERSION: int = 2

_HEADER_STRUCT = struct.Struct(">4sBBHQQII")
_IMAGE_PREFIX_STRUCT = struct.Struct(">BI")  # role:uint8 + length:uint32


class Frame(BaseModel):
    """
    One WebSocket message = one Frame.

    Wire format v2 (big-endian):
      header:
        4s   magic = b"SP02"
        u8   version = 2
        u8   flags (unused, 0)
        u16  reserved (0)
        u64  frame_id
        u64  send_timestamp_ns
        u32  meta_len  (UTF-8 JSON bytes)
        u32  image_count
      meta:
        meta_len bytes UTF-8 JSON
      images:
        repeated image_count times:
          u8   role (ImageRole)
          u32  jpeg_len
          jpeg_len bytes
    """

    MAGIC: ClassVar[bytes] = MAGIC
    VERSION: ClassVar[int] = VERSION

    frame_id: int = Field(ge=0)
    send_timestamp_ns: int = Field(ge=0)
    meta: dict[str, Any]
    images: dict[ImageRole, bytes]

    def to_wire(self) -> bytes:
        meta_bytes = json.dumps(self.meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        items = list(self.images.items())
        header = _HEADER_STRUCT.pack(
            self.MAGIC,
            self.VERSION,
            0,
            0,
            int(self.frame_id),
            int(self.send_timestamp_ns),
            len(meta_bytes),
            len(items),
        )

        out = bytearray()
        out.extend(header)
        out.extend(meta_bytes)

        for role, jpeg in items:
            out.extend(_IMAGE_PREFIX_STRUCT.pack(int(role), len(jpeg)))
            out.extend(jpeg)
        return bytes(out)

    @classmethod
    def from_wire(cls, payload: bytes) -> "Frame":
        if len(payload) < _HEADER_STRUCT.size:
            raise ValueError("payload too small for header")

        (
            magic,
            version,
            _flags,
            _reserved,
            frame_id,
            send_timestamp_ns,
            meta_len,
            image_count,
        ) = _HEADER_STRUCT.unpack_from(payload, 0)

        if magic != cls.MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if version != cls.VERSION:
            raise ValueError(f"unsupported version: {version}")

        offset = _HEADER_STRUCT.size
        if len(payload) < offset + meta_len:
            raise ValueError("payload too small for meta")

        meta_bytes = payload[offset : offset + meta_len]
        offset += meta_len
        meta = json.loads(meta_bytes.decode("utf-8"))

        images: dict[ImageRole, bytes] = {}
        for _ in range(int(image_count)):
            if len(payload) < offset + _IMAGE_PREFIX_STRUCT.size:
                raise ValueError("payload too small for image prefix")
            role_u8, jpeg_len = _IMAGE_PREFIX_STRUCT.unpack_from(payload, offset)
            offset += _IMAGE_PREFIX_STRUCT.size
            if len(payload) < offset + jpeg_len:
                raise ValueError("payload too small for image bytes")
            jpeg = payload[offset : offset + jpeg_len]
            offset += jpeg_len

            try:
                role = ImageRole(int(role_u8))
            except ValueError as e:
                raise ValueError(f"unknown ImageRole: {role_u8}") from e
            images[role] = jpeg

        if offset != len(payload):
            raise ValueError(f"payload length mismatch: got {len(payload)}, parsed {offset}")

        return cls(
            frame_id=int(frame_id),
            send_timestamp_ns=int(send_timestamp_ns),
            meta=meta,
            images=images,
        )
