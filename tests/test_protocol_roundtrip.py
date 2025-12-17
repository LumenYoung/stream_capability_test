import pytest

from streaming_protocol.models import StreamState
from streaming_protocol.protocol import decode_frame, encode_frame


def _make_good_meta(timestamp_ns: int = 123) -> dict:
    return {
        "state": [float(i) for i in range(50)],
        "remaining_action_chunks": [[float(j) for j in range(22)] for _ in range(50)],
        "timestamp_ns": int(timestamp_ns),
    }


def test_encode_decode_roundtrip_preserves_lengths_and_ids():
    meta = _make_good_meta(123)
    jpeg1 = b"\xff\xd8JPEG1\xff\xd9"
    jpeg2 = b"\xff\xd8JPEG2\xff\xd9"
    jpeg3 = b"\xff\xd8JPEG3\xff\xd9"

    payload = encode_frame(
        frame_id=42,
        send_timestamp_ns=123,
        meta=meta,
        jpeg1=jpeg1,
        jpeg2=jpeg2,
        jpeg3=jpeg3,
    )
    decoded = decode_frame(payload)

    assert decoded.frame_id == 42
    assert decoded.send_timestamp_ns == 123
    assert decoded.jpeg1 == jpeg1
    assert decoded.jpeg2 == jpeg2
    assert decoded.jpeg3 == jpeg3

    validated = StreamState.model_validate(decoded.meta)
    assert len(validated.state) == 50
    assert len(validated.remaining_action_chunks) == 50
    assert all(len(row) == 22 for row in validated.remaining_action_chunks)


def test_decode_rejects_bad_payload_length():
    meta = _make_good_meta(1)
    payload = encode_frame(frame_id=1, send_timestamp_ns=1, meta=meta, jpeg1=b"a", jpeg2=b"b", jpeg3=b"c")
    with pytest.raises(ValueError, match="payload length mismatch"):
        decode_frame(payload + b"junk")


def test_model_rejects_wrong_inner_dim():
    bad = {
        "state": [0.0] * 50,
        "remaining_action_chunks": [[0.0] * 21 for _ in range(50)],
        "timestamp_ns": 0,
    }
    with pytest.raises(Exception):
        StreamState.model_validate(bad)

