"""Public API for the c3po client library.

Provides :class:`Robot` — a single-object entry point that discovers
an N‑Droids robot station and exposes a Gym‑compatible ``reset()`` /
``step()`` interface.

Usage::

    from c3po import Robot

    with Robot() as robot:
        obs = robot.reset()
        while True:
            action = policy(obs)
            obs = robot.step(action)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from c3po._buffer import ObservationBuffer
from c3po._manifest import build_observation_keys, parse_manifest
from c3po._transport._websocket import WebSocketTransport
from c3po.exceptions import (
    IncompatibleVersionError,
    RobotDisconnectedError,
    RobotNotFoundError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "10.42.0.1"
_DEFAULT_PORT = 9090
_DISCOVERY_TIMEOUT = 10.0  # seconds


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------


class Robot:
    """Client for an N‑Droids robot station.

    Discovers the station via the ``/describe`` service over WebSocket,
    auto‑configures observation and action dictionaries from the station
    manifest, and provides a Gym‑compatible ``reset()`` / ``step()`` API.

    Parameters
    ----------
    rate:
        Desired step rate in Hz.  If ``None`` (default), the station's
        native ``control_rate`` is used.
    host:
        r2d2 server IP address.  Defaults to ``10.42.0.1`` (the DHCP
        server's static IP on the inference NIC).
    port:
        rosbridge WebSocket port.  Defaults to ``9090``.
    """

    def __init__(
        self,
        *,
        rate: float | None = None,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self._closed = False
        self._last_step_time: float | None = None

        # -- connect transport ---------------------------------------------------
        url = f"ws://{host}:{port}"
        self._transport = WebSocketTransport(url)
        try:
            self._transport.connect()
        except RobotDisconnectedError:
            raise RobotNotFoundError(
                f"Could not connect to r2d2 at {url}. Is the Ethernet cable plugged in?"
            )

        # -- discovery -----------------------------------------------------------
        try:
            raw = self._transport.call_service("/describe")
        except RobotDisconnectedError:
            self._transport.close()
            raise RobotNotFoundError(
                "Connected to rosbridge but /describe service call failed. "
                "Is r2d2 running?"
            )

        # -- parse manifest ------------------------------------------------------
        self._manifest = parse_manifest(raw)
        self._obs_keys = build_observation_keys(self._manifest)

        # -- version check -------------------------------------------------------
        api_version = self._manifest["api_version"]
        if not api_version.startswith("1."):
            raise IncompatibleVersionError(
                f"r2d2 reports API version {api_version}, but c3po requires version 1.x"
            )

        # -- step rate -----------------------------------------------------------
        if rate is not None:
            self.rate = float(rate)
        else:
            self.rate = self._manifest["control_rate"]

        # -- observation buffer --------------------------------------------------
        self._buffer = ObservationBuffer()

        # -- subscribe to observation topics -------------------------------------
        self._subscribe_all()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def station_model(self) -> str:
        """Station identifier (e.g. ``"so101_teleop"``)."""
        return self._manifest["station_model"]

    @property
    def control_rate(self) -> float:
        """Station's native control rate in Hz."""
        return self._manifest["control_rate"]

    @property
    def arms(self) -> list[str]:
        """Arm names.  Empty for stations with no actuated arms."""
        return list(self._manifest["arm_names"])

    @property
    def cameras(self) -> list[str]:
        """Camera key names present in observations."""
        return list(self._manifest["camera_names"])

    @property
    def controllers(self) -> list[str]:
        """Controller key names present in observations."""
        return list(self._manifest["controller_names"])

    @property
    def action_keys(self) -> list[str]:
        """Keys that :meth:`step` expects in the action dictionary."""
        return list(self._manifest["action_keys"])

    @property
    def joint_names(self) -> dict[str, list[str]]:
        """Mapping from arm key to ordered joint name list."""
        return dict(self._manifest["joint_names"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        """Return the current observation snapshot.

        Resets the step‑rate limiter.  Call once at the start of
        each episode.

        Blocks until at least one observation has been received from
        every subscribed topic.
        """
        if self._closed:
            raise RobotDisconnectedError("Robot is closed.")

        # Wait for initial data on all expected topics.
        deadline = time.monotonic() + _DISCOVERY_TIMEOUT
        while True:
            snapshot = self._buffer.get()
            missing = set(self._obs_keys) - set(snapshot)
            if not missing:
                self._last_step_time = time.monotonic()
                return snapshot
            if time.monotonic() > deadline:
                raise RobotDisconnectedError(
                    f"Timeout waiting for initial observations. Missing: {missing}"
                )
            time.sleep(0.01)

    def step(
        self, action: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        """Send *action* to the robot, then return the latest observation.

        Paces the control loop at :attr:`rate` Hz.  If the policy takes
        longer than one period, returns immediately.  The observation is
        always the most recent data from each sensor — it never blocks
        waiting for a slow sensor to produce new data.

        Parameters
        ----------
        action:
            Dictionary mapping action keys to NumPy arrays.  Pass
            ``None`` for a read‑only step (no command sent).

        Returns
        -------
        dict
            Latest observation snapshot.
        """
        if self._closed:
            raise RobotDisconnectedError("Robot is closed.")

        # -- rate limiting ---------------------------------------------------
        if self._last_step_time is not None:
            elapsed = time.monotonic() - self._last_step_time
            period = 1.0 / self.rate
            if elapsed < period:
                time.sleep(period - elapsed)

        # -- send action -----------------------------------------------------
        if action is not None:
            self._validate_action(action)
            self._publish_action(action)

        # -- gather observation ----------------------------------------------
        self._last_step_time = time.monotonic()
        return self._buffer.get()

    def close(self) -> None:
        """Disconnect from the station and release resources."""
        if self._closed:
            return
        self._closed = True
        try:
            self._transport.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Robot:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal: subscriptions
    # ------------------------------------------------------------------

    def _subscribe_all(self) -> None:
        """Create subscriptions for every key in the observation set."""
        # -- joint state subscriptions ---------------------------------------
        for arm_info in self._manifest["arms"]:
            name = arm_info["name"]
            topic_prefix = f"/observations/{name}/" if name else "/observations/"
            self._transport.subscribe(
                topic_prefix + "joint_position",
                "sensor_msgs/JointState",
                self._make_joint_callback(name),
            )

        # -- camera subscriptions --------------------------------------------
        for cam_info in self._manifest["cameras"]:
            cam_name = cam_info["name"]
            topic = f"/observations/{cam_name}"
            self._transport.subscribe(
                topic,
                "sensor_msgs/Image",
                self._make_image_callback(cam_name),
            )

        # -- controller subscriptions ----------------------------------------
        for ctrl_info in self._manifest["controllers"]:
            ctrl_name = ctrl_info["name"]
            ctrl_type = ctrl_info.get("type", "")
            topic = f"/controllers/{ctrl_name}/state"
            if ctrl_type == "joint_position":
                self._transport.subscribe(
                    topic,
                    "sensor_msgs/JointState",
                    self._make_controller_callback(ctrl_name),
                )
            else:
                self._transport.subscribe(
                    topic,
                    "std_msgs/Float64MultiArray",
                    self._make_controller_callback(ctrl_name),
                )

    # ------------------------------------------------------------------
    # Internal: callbacks
    # ------------------------------------------------------------------

    def _make_joint_callback(self, arm_name: str):
        prefix = f"{arm_name}/" if arm_name else ""

        def callback(msg: dict[str, Any]) -> None:
            pos = np.array(msg.get("position", []), dtype=np.float64)
            vel = np.array(msg.get("velocity", []), dtype=np.float64)
            self._buffer.update(f"{prefix}joint_position", pos)
            self._buffer.update(f"{prefix}joint_velocity", vel)

        return callback

    def _make_image_callback(self, cam_name: str):
        def callback(msg: dict[str, Any]) -> None:
            height = msg.get("height", 0)
            width = msg.get("width", 0)
            encoding = msg.get("encoding", "")
            data = msg.get("data", [])

            if encoding in ("rgb8", "bgr8", "8UC3"):
                arr = np.array(data, dtype=np.uint8).reshape((height, width, 3))
                if encoding == "bgr8":
                    arr = arr[..., ::-1]  # BGR → RGB
            elif encoding == "16UC1":
                arr = np.array(data, dtype=np.uint16).reshape((height, width))
            elif encoding == "mono8":
                arr = np.array(data, dtype=np.uint8).reshape((height, width))
            else:
                arr = np.array(data, dtype=np.uint8).reshape((height, width, -1))

            self._buffer.update(cam_name, arr)

        return callback

    def _make_controller_callback(self, ctrl_name: str):
        def callback(msg: dict[str, Any]) -> None:
            # Try JointState format first, then generic.
            if "position" in msg:
                pos = np.array(msg["position"], dtype=np.float64)
                self._buffer.update(f"{ctrl_name}/joint_position", pos)
            elif "data" in msg:
                arr = np.array(msg["data"], dtype=np.float64)
                self._buffer.update(f"{ctrl_name}/state", arr)

        return callback

    # ------------------------------------------------------------------
    # Internal: validation and publishing
    # ------------------------------------------------------------------

    def _validate_action(self, action: dict[str, np.ndarray]) -> None:
        """Check that *action* contains all required keys."""
        missing = set(self.action_keys) - set(action)
        if missing:
            raise ValueError(
                f"Action dict missing required keys: {missing}. "
                f"Expected keys: {self.action_keys}"
            )

    def _publish_action(self, action: dict[str, np.ndarray]) -> None:
        """Publish each action key to the corresponding command topic."""
        for key, value in action.items():
            # Determine the arm name and command mode from the key.
            # Key format: "{arm}/{mode}" or "{mode}" for single-arm.
            if "/" in key:
                arm_name, mode = key.rsplit("/", 1)
                topic = f"/commands/{arm_name}/{mode}"
            else:
                mode = key
                topic = f"/commands/{mode}"

            msg = {
                "mode": mode,
                "joint_names": self.joint_names.get(
                    arm_name if "/" in key else "default", []
                ),
                "data": value.astype(np.float64).ravel().tolist(),
            }
            self._transport.publish(topic, "c3po_msgs/Action", msg)
