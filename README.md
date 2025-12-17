# streaming_protocol (PoC)

WebSocket-based JPEG streaming proof-of-concept with a small custom binary protocol:

- Server: loads a video, decodes all frames, resizes to `480x300`, JPEG-encodes, preloads into RAM, and streams 3 JPEGs + metadata per message.
- Client: receives messages, parses metadata (Pydantic), converts image + state to numpy arrays (not used yet), and exposes a tiny webpage to prove reception + show latency.

## Run

### Start server

The server loads a video at startup, pre-decodes it, resizes to `480x300`, JPEG-encodes, and keeps all frames in RAM. It then streams at the configured FPS (e.g. `30`).

```bash
python -m streaming_protocol.server --host 127.0.0.1 --port 8765 --fps 30 path/to/video.mp4
```

### Start client + viewer

The client connects to the WebSocket, decodes each message, computes latency as:

`latency_ms = (client_receive_time_ns - server_send_time_ns) / 1e6`

It exposes a small HTTP page that shows the latest received JPEG (only the first image for now) plus metadata + latency.

```bash
python -m streaming_protocol.client --ws ws://127.0.0.1:8765 --http-port 8000
```

Open `http://127.0.0.1:8000`.

### Notes

- The “send timestamp” is currently a wall-clock Unix timestamp in nanoseconds via `time.time_ns()` on the server.
- Each message contains 3 JPEGs + metadata, but the webpage renders only `jpeg1` as a proof-of-receiving.

## Protocol

Each WebSocket binary message contains:

- header (magic/version/frame_id/timestamps + lengths)
- metadata JSON (`{"state":[...50 floats...],"timestamp_ns":...}`)
- 3 JPEG byte blobs (each `480x300`)
