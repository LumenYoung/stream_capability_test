from __future__ import annotations

import argparse
import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web
import cv2
import numpy as np
import websockets

from streaming_protocol.models import StreamState
from streaming_protocol.protocol import Frame, ImageRole


@dataclass
class LatestFrame:
    frame_id: int = -1
    send_timestamp_ns: int = 0
    recv_timestamp_ns: int = 0
    latency_ms: float = 0.0
    jpeg1: bytes = b""
    meta: dict[str, Any] | None = None


def _ns_now() -> int:
    return time.time_ns()


def _jpeg_bytes_to_bgr(jpeg: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img


async def ws_receiver_task(ws_url: str, latest: LatestFrame) -> None:
    async with websockets.connect(ws_url, max_size=None) as ws:
        async for message in ws:
            if not isinstance(message, (bytes, bytearray)):
                continue

            recv_ts = _ns_now()
            frame = Frame.from_wire(bytes(message))
            meta = frame.meta
            jpeg1 = frame.images.get(ImageRole.LEFT) or b""

            # Convert to numpy arrays (not used yet)
            if jpeg1:
                _img1 = _jpeg_bytes_to_bgr(jpeg1)
            _state = np.asarray(meta.state, dtype=np.float32)
            _remaining_action_chunks = np.asarray(meta.remaining_action_chunks, dtype=np.float32)

            latency_ms = (recv_ts - meta.timestamp_ns) / 1e6
            latest.frame_id = frame.frame_id
            latest.send_timestamp_ns = meta.timestamp_ns
            latest.recv_timestamp_ns = recv_ts
            latest.latency_ms = float(latency_ms)
            latest.jpeg1 = jpeg1
            latest.meta = meta.model_dump()


def _jpeg_data_url(jpeg: bytes) -> str:
    b64 = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


async def http_server_task(host: str, port: int, latest: LatestFrame) -> None:
    async def index(_request: web.Request) -> web.Response:
        html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Streaming PoC</title>
    <style>
      body { font-family: sans-serif; padding: 16px; }
      .row { display: flex; gap: 16px; align-items: flex-start; }
      img { border: 1px solid #ddd; width: 480px; height: 300px; object-fit: contain; }
      pre { background: #f7f7f7; padding: 12px; border: 1px solid #eee; width: 480px; overflow: auto; }
      .kpi { margin: 8px 0 16px; }
      .kpi span { display: inline-block; min-width: 180px; }
    </style>
  </head>
  <body>
    <h2>WebSocket JPEG Streaming PoC</h2>
    <div class="kpi">
      <span>frame_id: <b id="frame_id">-</b></span>
      <span>latency_ms: <b id="latency_ms">-</b></span>
    </div>
    <div class="row">
      <img id="img" alt="stream" />
      <pre id="meta">{}</pre>
    </div>
    <script>
      async function poll() {
        try {
          const r = await fetch('/latest.json', { cache: 'no-store' });
          const j = await r.json();
          document.getElementById('frame_id').textContent = j.frame_id;
          document.getElementById('latency_ms').textContent = j.latency_ms.toFixed(2);
          document.getElementById('img').src = j.jpeg1_data_url;
          document.getElementById('meta').textContent = JSON.stringify(j.meta, null, 2);
        } catch (e) {
          // ignore
        }
        setTimeout(poll, 50);
      }
      poll();
    </script>
  </body>
</html>
"""
        return web.Response(text=html, content_type="text/html")

    async def latest_json(_request: web.Request) -> web.Response:
        if latest.frame_id < 0:
            return web.json_response({"ok": False})
        return web.json_response(
            {
                "ok": True,
                "frame_id": latest.frame_id,
                "send_timestamp_ns": latest.send_timestamp_ns,
                "recv_timestamp_ns": latest.recv_timestamp_ns,
                "latency_ms": latest.latency_ms,
                "jpeg1_data_url": _jpeg_data_url(latest.jpeg1),
                "meta": latest.meta,
            }
        )

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/latest.json", latest_json)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"http://{host}:{port} viewer")

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WebSocket client + web viewer (PoC)")
    p.add_argument("--ws", default="ws://127.0.0.1:8765")
    p.add_argument("--http-host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=8000)
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    latest = LatestFrame()

    await asyncio.gather(
        ws_receiver_task(args.ws, latest),
        http_server_task(args.http_host, args.http_port, latest),
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
