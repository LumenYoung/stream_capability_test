import pytest

from streaming_protocol.models import StreamState
from streaming_protocol.protocol import Frame, ImageRole


def _make_good_meta(timestamp_ns: int = 123) -> dict:
    return {
        "state": [float(i) for i in range(50)],
        "remaining_action_chunks": [[float(j) for j in range(22)] for _ in range(50)],
        "timestamp_ns": int(timestamp_ns),
    }


def test_frame_to_wire_from_wire_roundtrip():
    meta = _make_good_meta(123)
    jpeg_left = b"\xff\xd8LEFT\xff\xd9"
    jpeg_center = b"\xff\xd8CENTER\xff\xd9"
    jpeg_right = b"\xff\xd8RIGHT\xff\xd9"
    jpeg_back = b"\xff\xd8BACK\xff\xd9"

    frame = Frame(
        frame_id=42,
        send_timestamp_ns=123,
        meta=meta,
        images={
            ImageRole.LEFT: jpeg_left,
            ImageRole.CENTER: jpeg_center,
            ImageRole.RIGHT: jpeg_right,
            ImageRole.BACK: jpeg_back,
        },
    )
    payload = frame.to_wire()
    decoded = Frame.from_wire(payload)

    assert decoded.frame_id == 42
    assert decoded.send_timestamp_ns == 123
    assert decoded.images[ImageRole.LEFT] == jpeg_left
    assert decoded.images[ImageRole.CENTER] == jpeg_center
    assert decoded.images[ImageRole.RIGHT] == jpeg_right
    assert decoded.images[ImageRole.BACK] == jpeg_back

    validated = StreamState.model_validate(decoded.meta)
    assert len(validated.state) == 50
    assert len(validated.remaining_action_chunks) == 50
    assert all(len(row) == 22 for row in validated.remaining_action_chunks)


def test_from_wire_rejects_trailing_bytes():
    meta = _make_good_meta(1)
    frame = Frame(
        frame_id=1,
        send_timestamp_ns=1,
        meta=meta,
        images={ImageRole.LEFT: b"a"},
    )
    payload = frame.to_wire()
    with pytest.raises(ValueError, match="payload length mismatch"):
        Frame.from_wire(payload + b"junk")


def test_from_wire_rejects_unknown_image_role():
    meta = _make_good_meta(1)
    frame = Frame(
        frame_id=1,
        send_timestamp_ns=1,
        meta=meta,
        images={ImageRole.LEFT: b"a"},
    )
    payload = frame.to_wire()

    # Find the role byte: header + meta, then role byte.
    # Avoid mutating bytes inside the meta JSON; compute the exact offset.
    mutated = bytearray(payload)
    import json

    meta_len = len(json.dumps(frame.meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    role_offset = 32 + meta_len  # header is 32 bytes in v2
    mutated[role_offset] = 255

    with pytest.raises(ValueError, match="unknown ImageRole"):
        Frame.from_wire(bytes(mutated))


def test_model_rejects_wrong_inner_dim_strict_22():
    bad = {
        "state": [0.0] * 50,
        "remaining_action_chunks": [[0.0] * 21 for _ in range(50)],
        "timestamp_ns": 0,
    }
    with pytest.raises(Exception):
        StreamState.model_validate(bad)
