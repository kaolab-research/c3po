"""Shared test fixtures for c3po.

Provides a mock r2d2 WebSocket server that speaks the N-Droids custom protocol.
Each test gets a fresh server instance on a random port with synthetic data.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import suppress

import numpy as np
import pytest
import websockets
import websockets.asyncio.server
from PIL import Image

from c3po._protocol import (
    BinaryFrame,
    CancelEpisode,
    DescribeRequest,
    DescribeResponse,
    Encoding,
    Observation,
    StartRecording,
    StopRecording,
    decode_json_message,
    encode_binary_frame,
    encode_json_message,
)

# ---------------------------------------------------------------------------
# Mock station manifest — a single-arm SO-101 with a leader and one camera
# ---------------------------------------------------------------------------

MOCK_MANIFEST: dict = {
    "api_version": "1.0",
    "station_model": "so101_teleop",
    "control_rate": 50.0,
    "arms": [
        {
            "name": "follower",
            "joint_count": 6,
            "joint_names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
            "command_mode": "joint_position",
        }
    ],
    "cameras": [
        {"name": "front_rgb", "resolution": [720, 1280]},
    ],
    "controllers": [
        {
            "name": "leader",
            "type": "joint_position",
            "joint_count": 6,
            "joint_names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
        }
    ],
}

# A bimanual ALOHA manifest for testing multi-arm stations.
MOCK_MANIFEST_BIMANUAL: dict = {
    "api_version": "1.0",
    "station_model": "aloha",
    "control_rate": 50.0,
    "arms": [
        {
            "name": "left_follower",
            "joint_count": 6,
            "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
            "command_mode": "joint_position",
        },
        {
            "name": "right_follower",
            "joint_count": 6,
            "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
            "command_mode": "joint_position",
        },
    ],
    "cameras": [
        {"name": "left_wrist_rgb", "resolution": [480, 640]},
        {"name": "right_wrist_rgb", "resolution": [480, 640]},
        {"name": "front_rgb", "resolution": [720, 1280]},
    ],
    "controllers": [
        {
            "name": "left_leader",
            "type": "joint_position",
            "joint_count": 6,
            "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
        },
        {
            "name": "right_leader",
            "type": "joint_position",
            "joint_count": 6,
            "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_joint_positions(t: float, n: int = 6) -> list[float]:
    """Synthetic sinusoidal joint positions."""
    return [float(np.sin(t + i * 0.7)) for i in range(n)]


def _make_joint_velocities(t: float, n: int = 6) -> list[float]:
    """Synthetic joint velocities (derivative of sin)."""
    return [float(np.cos(t + i * 0.7)) for i in range(n)]


def _make_jpeg_frame() -> bytes:
    """A tiny 64×64 random-noise JPEG frame for testing."""
    rng = np.random.default_rng()
    img = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mock r2d2 protocol handler
# ---------------------------------------------------------------------------


async def _r2d2_handler(
    websocket, *, manifest: dict = MOCK_MANIFEST, control_rate: float = 50.0
):
    """Handle a single c3po client connection, speaking the N-Droids protocol."""
    stream_task: asyncio.Task | None = None
    _recording = False
    _last_action: dict | None = None

    async def _stream_loop() -> None:
        """Send observations and binary frames at *control_rate* Hz."""
        t = 0.0
        period = 1.0 / control_rate
        camera_names = [c["name"] for c in manifest.get("cameras", [])]
        while True:
            await asyncio.sleep(period)
            t += period

            # Build observation: joint state from all arms and controllers.
            obs_data: dict[str, list[float]] = {}
            for arm in manifest.get("arms", []):
                name = arm["name"]
                prefix = f"{name}/" if name else ""
                n = arm.get("joint_count", 6)
                obs_data[f"{prefix}joint_position"] = _make_joint_positions(t, n)
                obs_data[f"{prefix}joint_velocity"] = _make_joint_velocities(t, n)
            for ctrl in manifest.get("controllers", []):
                n = ctrl.get("joint_count", 6)
                obs_data[f"{ctrl['name']}/joint_position"] = _make_joint_positions(
                    t + 0.5, n
                )

            obs_msg = Observation(data=obs_data)
            try:
                await websocket.send(encode_json_message(obs_msg))
            except Exception:
                break

            # Send a binary JPEG frame for each camera.
            for cam_name in camera_names:
                jpeg = _make_jpeg_frame()
                ts_ns = int(t * 1e9)
                # Resolution from manifest, default to 64×64 for test images.
                cam_info = next(
                    (c for c in manifest["cameras"] if c["name"] == cam_name), None
                )
                w, h = (64, 64)
                if cam_info:
                    res = cam_info.get("resolution", [64, 64])
                    h, w = res[0], res[1]

                frame = BinaryFrame(
                    camera_name=cam_name,
                    timestamp_ns=ts_ns,
                    width=w,
                    height=h,
                    encoding=Encoding.JPEG_RGB,
                    jpeg_bytes=jpeg,
                )
                try:
                    await websocket.send(encode_binary_frame(frame))
                except Exception:
                    break

    try:
        async for raw in websocket:
            if isinstance(raw, bytes):
                # Binary frames from client?  Not expected in this version; ignore.
                continue

            try:
                msg = decode_json_message(raw)
            except Exception:
                continue

            # Describe: send manifest, then start streaming.
            if isinstance(msg, DescribeRequest):
                response = DescribeResponse(manifest=manifest)
                await websocket.send(encode_json_message(response))
                if stream_task is None:
                    stream_task = asyncio.create_task(_stream_loop())

            elif isinstance(msg, StartRecording):
                _recording = True
            elif isinstance(msg, StopRecording):
                _recording = False
            elif isinstance(msg, CancelEpisode):
                pass  # acknowledge silently
            elif hasattr(msg, "data"):
                # Action message — store for test assertions
                _last_action = msg.data

    finally:
        if stream_task is not None:
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task


# ---------------------------------------------------------------------------
# Mock r2d2 server (runs in a background thread with asyncio)
# ---------------------------------------------------------------------------


class MockR2D2Server:
    """A mock r2d2 server that speaks the N-Droids custom protocol.

    Runs a WebSocket server in a background thread on a random port.
    Each test gets its own instance — no cross-test state leakage.

    Parameters
    ----------
    host:
        Bind address.  Defaults to ``127.0.0.1``.
    port:
        Bind port.  Defaults to ``0`` (OS picks a free port).
    manifest:
        Station manifest to serve in the ``describe_response``.
    control_rate:
        Streaming rate in Hz for synthetic observations and camera frames.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        manifest: dict | None = None,
        control_rate: float = 50.0,
    ) -> None:
        self.host = host
        self.port = port
        self.manifest = manifest if manifest is not None else MOCK_MANIFEST
        self.control_rate = control_rate
        self._thread: threading.Thread | None = None
        self._server = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        """Start the server in a background daemon thread."""

        async def _run() -> None:
            self._server = await websockets.asyncio.server.serve(
                lambda ws: _r2d2_handler(
                    ws, manifest=self.manifest, control_rate=self.control_rate
                ),
                self.host,
                self.port,
            )
            # Read back the actual port (important when port=0).
            for sock in self._server.sockets:
                if hasattr(sock, "getsockname"):
                    addr = sock.getsockname()
                    if len(addr) >= 2:
                        self.port = addr[1]
            self._ready.set()
            await self._server.serve_forever()

        def _thread_target() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(_run())
            except (asyncio.CancelledError, RuntimeError):
                pass
            finally:
                with suppress(Exception):
                    self._loop.close()

        self._thread = threading.Thread(target=_thread_target, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the server and clean up the event loop."""
        if (
            self._server is not None
            and self._loop is not None
            and self._loop.is_running()
        ):
            self._loop.call_soon_threadsafe(self._server.close)
        time.sleep(0.05)
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_server() -> MockR2D2Server:
    """Per-test fixture: a fresh mock r2d2 server on a random port.

    Each test gets its own server instance.  The server is automatically
    stopped after the test, even on failure.
    """
    server = MockR2D2Server()
    server.start()
    try:
        yield server
    finally:
        server.stop()
