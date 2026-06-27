"""Shared test fixtures for c3po.

Provides a mock rosbridge WebSocket server that simulates an r2d2
station with a single SO-101 arm and a leader controller.
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

# ---------------------------------------------------------------------------
# Mock station manifest
# ---------------------------------------------------------------------------

MOCK_MANIFEST = {
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
        {"name": "front_rgb", "resolution": [1280, 720]},
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

MOCK_DESCRIBE_RESPONSE = {
    "robot_model": "so101",
    "joint_count": 6,
    "control_rate": 50.0,
    "camera_names": ["front_rgb"],
    "command_mode": "joint_position",
    "joint_names": [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ],
    "manifest_json": json.dumps(MOCK_MANIFEST),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_joint_state_msg(t: float, n: int = 6) -> dict:
    return {
        "header": {"stamp": {"sec": int(t), "nanosec": 0}, "frame_id": ""},
        "name": [f"joint_{i}" for i in range(n)],
        "position": [float(np.sin(t + i)) for i in range(n)],
        "velocity": [float(np.cos(t + i)) for i in range(n)],
        "effort": [],
    }


def _make_image_msg(t: float) -> dict:
    rng = np.random.RandomState(int(t * 1000))
    img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return {
        "header": {"stamp": {"sec": int(t), "nanosec": 0}, "frame_id": ""},
        "height": 64,
        "width": 64,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 192,
        "data": img.ravel().tolist(),
    }


# ---------------------------------------------------------------------------
# Mock rosbridge server (function‑scoped, self‑cleaning)
# ---------------------------------------------------------------------------


async def _rosbridge_handler(websocket):
    """Handle a single rosbridge client connection."""
    subscriptions: dict[str, str] = {}
    publish_task: asyncio.Task | None = None

    async def _publish_loop():
        t = 0.0
        while True:
            await asyncio.sleep(0.05)
            t += 0.05
            for topic in list(subscriptions):
                try:
                    if "joint_position" in topic and "/observations/" in topic:
                        await websocket.send(
                            json.dumps(
                                {
                                    "op": "publish",
                                    "topic": topic,
                                    "msg": _make_joint_state_msg(t),
                                }
                            )
                        )
                    elif "_rgb" in topic and "/observations/" in topic:
                        await websocket.send(
                            json.dumps(
                                {
                                    "op": "publish",
                                    "topic": topic,
                                    "msg": _make_image_msg(t),
                                }
                            )
                        )
                    elif topic.startswith("/controllers/"):
                        await websocket.send(
                            json.dumps(
                                {
                                    "op": "publish",
                                    "topic": topic,
                                    "msg": _make_joint_state_msg(t + 0.5),
                                }
                            )
                        )
                except Exception:
                    break

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            op = msg.get("op")

            if op == "call_service":
                sid = msg.get("id", "")
                service = msg.get("service", "")
                if service == "/describe":
                    await websocket.send(
                        json.dumps(
                            {
                                "op": "service_response",
                                "id": sid,
                                "service": service,
                                "values": MOCK_DESCRIBE_RESPONSE,
                            }
                        )
                    )
                else:
                    await websocket.send(
                        json.dumps(
                            {
                                "op": "service_response",
                                "id": sid,
                                "service": service,
                                "values": {},
                            }
                        )
                    )

            elif op == "subscribe":
                topic = msg.get("topic", "")
                subscriptions[topic] = msg.get("type", "")
                if publish_task is None:
                    publish_task = asyncio.create_task(_publish_loop())

            elif op == "unsubscribe":
                subscriptions.pop(msg.get("topic", ""), None)

            elif op == "publish":
                pass  # command — ack silently

    finally:
        if publish_task is not None:
            publish_task.cancel()
            with suppress(asyncio.CancelledError):
                await publish_task


class MockRosbridgeServer:
    """Runs a mock rosbridge WebSocket server in a background thread.

    Use as a context manager or with explicit start()/stop().
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None
        self._server = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        """Start the server in a background thread."""

        async def _run():
            self._server = await websockets.asyncio.server.serve(
                _rosbridge_handler, self.host, self.port
            )
            for sock in self._server.sockets:
                if hasattr(sock, "getsockname"):
                    addr = sock.getsockname()
                    if len(addr) >= 2:
                        self.port = addr[1]
            self._ready.set()
            await self._server.serve_forever()

        def _thread_target():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(_run())
            except (asyncio.CancelledError, RuntimeError):
                pass  # expected during teardown
            finally:
                with suppress(Exception):
                    self._loop.close()

        self._thread = threading.Thread(target=_thread_target, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the server and clean up."""
        # Close the server from the event loop thread.
        if (
            self._server is not None
            and self._loop is not None
            and self._loop.is_running()
        ):
            self._loop.call_soon_threadsafe(self._server.close)
        # Give it a moment to clean up, then stop the loop.
        time.sleep(0.05)
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_server():
    """Per‑test fixture: a fresh mock rosbridge server.

    Each test gets its own server instance to avoid cross‑test state
    leakage and teardown complexity.
    """
    server = MockRosbridgeServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()
