from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np
import websockets

from streaming_protocol.models import StreamState
from streaming_protocol.protocol import Frame, ImageRole


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    fps: float
    width: int
    height: int
    jpeg_quality: int
    video_path: str


def _ns_now() -> int:
    return time.time_ns()


def _rand_state() -> list[float]:
    return [random.random() for _ in range(50)]


def _rand_remaining_action_chunks() -> list[list[float]]:
    n = random.randint(20, 50)
    return [[random.random() for _ in range(22)] for _ in range(n)]


def _resize_and_jpeg(bgr: np.ndarray, *, width: int, height: int, jpeg_quality: int) -> bytes:
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def preload_video_as_quads(cfg: ServerConfig) -> list[tuple[bytes, bytes, bytes, bytes]]:
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {cfg.video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < 4:
        raise RuntimeError(f"video must have at least 4 frames; got {len(frames)}")

    quads: list[tuple[bytes, bytes, bytes, bytes]] = []
    for i in range(0, len(frames) - 3):
        j1 = _resize_and_jpeg(frames[i], width=cfg.width, height=cfg.height, jpeg_quality=cfg.jpeg_quality)
        j2 = _resize_and_jpeg(frames[i + 1], width=cfg.width, height=cfg.height, jpeg_quality=cfg.jpeg_quality)
        j3 = _resize_and_jpeg(frames[i + 2], width=cfg.width, height=cfg.height, jpeg_quality=cfg.jpeg_quality)
        j4 = _resize_and_jpeg(frames[i + 3], width=cfg.width, height=cfg.height, jpeg_quality=cfg.jpeg_quality)
        quads.append((j1, j2, j3, j4))

    if not quads:
        raise RuntimeError("no quads generated from video")
    return quads


async def stream_handler(ws: websockets.ServerConnection, cfg: ServerConfig, quads: list[tuple[bytes, bytes, bytes, bytes]]):
    frame_id = 0
    next_idx = 0
    period_s = 1.0 / cfg.fps if cfg.fps > 0 else 0.0
    bytes_sent = 0
    frames_sent = 0
    last_report_ns = _ns_now()
    missed_deadlines = 0
    start_ns = _ns_now()
    dropped_frames = 0

    latest_payload: bytes | None = None
    latest_payload_bytes = 0
    latest_frame_id = -1

    # If the client is slower than the server, we prefer to drop frames by not awaiting too long.
    # websockets will still apply backpressure, but period pacing limits server send rate.
    while True:
        if period_s > 0:
            target_ns = start_ns + int(frame_id * period_s * 1e9)
            now_ns = _ns_now()
            if now_ns < target_ns:
                await asyncio.sleep((target_ns - now_ns) / 1e9)
            else:
                if now_ns - target_ns > int(0.5 * period_s * 1e9):
                    missed_deadlines += 1

        send_ts = _ns_now()
        meta = StreamState(
            state=_rand_state(),
            remaining_action_chunks=_rand_remaining_action_chunks(),
            timestamp_ns=send_ts,
        )
        jpeg1, jpeg2, jpeg3, jpeg4 = quads[next_idx]

        payload = Frame(
            frame_id=frame_id,
            send_timestamp_ns=send_ts,
            meta=meta,
            images={
                ImageRole.LEFT: jpeg1,
                ImageRole.CENTER: jpeg2,
                ImageRole.RIGHT: jpeg3,
                ImageRole.BACK: jpeg4,
            },
        ).to_wire()

        # "Always latest" behavior:
        # Keep only the newest payload; if the previous newest hasn't been sent yet, consider it dropped.
        if latest_payload is not None:
            dropped_frames += 1
        latest_payload = payload
        latest_payload_bytes = len(payload)
        latest_frame_id = frame_id

        # Attempt to flush the latest payload; if backpressured, keep it for a later attempt.
        if latest_payload is not None:
            try:
                await asyncio.wait_for(ws.send(latest_payload), timeout=0.002)
                bytes_sent += latest_payload_bytes
                frames_sent += 1
                latest_payload = None
                latest_payload_bytes = 0
                latest_frame_id = -1
            except TimeoutError:
                pass

        now_ns = _ns_now()
        if now_ns - last_report_ns >= 1_000_000_000:
            elapsed_s = (now_ns - last_report_ns) / 1e9
            mib_s = (bytes_sent / (1024 * 1024)) / elapsed_s
            fps = frames_sent / elapsed_s
            print(
                f"tx {mib_s:.2f} MiB/s ({bytes_sent} bytes), {fps:.1f} msg/s, "
                f"missed_deadlines={missed_deadlines}, dropped_frames={dropped_frames}"
            )
            bytes_sent = 0
            frames_sent = 0
            missed_deadlines = 0
            dropped_frames = 0
            last_report_ns = now_ns

        frame_id += 1
        next_idx = (next_idx + 1) % len(quads)


async def run_server(cfg: ServerConfig) -> None:
    quads = preload_video_as_quads(cfg)
    print(f"preloaded {len(quads)} quads into RAM")

    async def _handler(ws: websockets.ServerConnection):
        print("client connected")
        try:
            await stream_handler(ws, cfg, quads)
        except websockets.ConnectionClosed:
            pass
        finally:
            print("client disconnected")

    async with websockets.serve(_handler, cfg.host, cfg.port, max_size=None):
        print(f"ws://{cfg.host}:{cfg.port} streaming @ {cfg.fps} fps ({cfg.width}x{cfg.height})")
        await asyncio.Future()


def parse_args() -> ServerConfig:
    p = argparse.ArgumentParser(description="WebSocket JPEG triplet streamer (PoC)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=300)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("video", help="path to video file")
    a = p.parse_args()
    return ServerConfig(
        host=a.host,
        port=a.port,
        fps=a.fps,
        width=a.width,
        height=a.height,
        jpeg_quality=a.jpeg_quality,
        video_path=a.video,
    )


def main() -> None:
    cfg = parse_args()
    asyncio.run(run_server(cfg))


if __name__ == "__main__":
    main()
