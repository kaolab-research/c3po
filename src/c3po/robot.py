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
import time
from typing import Any

import numpy as np

from c3po._buffer import ObservationBuffer
from c3po._manifest import build_observation_keys, parse_manifest
from c3po._protocol import (
    Action,
    BinaryFrame,
    CancelEpisode,
    DescribeRequest,
    Observation,
    ResetEpisode,
    StartRecording,
    StopRecording,
)
from c3po._transport._websocket import WebSocketTransport
from c3po.exceptions import (
    IncompatibleVersionError,
    RobotDisconnectedError,
    RobotNotFoundError,
)

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "10.42.0.1"
_DEFAULT_PORT = 9090
_DISCOVERY_TIMEOUT = 10.0


class Robot:
    """Client for an N‑Droids robot station.

    Discovers the station via a ``describe`` handshake over WebSocket,
    auto‑configures observation and action dictionaries from the station
    manifest, and provides a Gym‑compatible ``reset()`` / ``step()`` API.

    r2d2 owns the control loop cadence.  ``step()`` blocks until the
    next observation arrives, so the researcher's loop is naturally
    paced at the station's native rate.

    Parameters
    ----------
    host:
        r2d2 server IP address.  Defaults to ``10.42.0.1``.
    port:
        WebSocket port.  Defaults to ``9090``.
    rate:
        Desired control rate in Hz.  Defaults to 50.0.
        r2d2 may cap this at the hardware's maximum safe rate.
    """

    def __init__(
        self,
        *,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        rate: float = 50.0,
    ) -> None:
        self._closed = False
        self._recording = False
        self._rate = float(rate)

        # -- connect transport --------------------------------------------------
        url = f"ws://{host}:{port}"
        self._transport = WebSocketTransport(url)
        try:
            self._transport.connect()
        except RobotDisconnectedError:
            raise RobotNotFoundError(
                f"Could not connect to r2d2 at {url}. "
                "Is the Ethernet cable plugged in and r2d2 running?"
            )

        # -- discover ------------------------------------------------------------
        try:
            self._transport.send_json(DescribeRequest())
            resp = self._transport.recv(timeout=_DISCOVERY_TIMEOUT)
        except RobotDisconnectedError:
            self._transport.close()
            raise RobotNotFoundError(
                "Connected but /describe handshake failed. Is r2d2 running?"
            )

        if not hasattr(resp, "manifest"):
            raise RobotNotFoundError(
                f"Expected describe_response, got {type(resp).__name__}"
            )

        # -- parse manifest ------------------------------------------------------
        self._manifest = parse_manifest(resp.manifest)
        self._obs_keys = build_observation_keys(self._manifest)

        # -- version check -------------------------------------------------------
        api_version = self._manifest["api_version"]
        if not api_version.startswith("1."):
            raise IncompatibleVersionError(
                f"r2d2 reports API version {api_version}, but c3po requires version 1.x"
            )

        # -- configure rate -------------------------------------------------------
        from c3po._protocol import Configure

        self._transport.send_json(Configure(rate=self._rate))

        # -- observation buffer --------------------------------------------------
        self._buffer = ObservationBuffer()

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
        return self._manifest.get("control_rate", self._rate)

    @property
    def rate(self) -> float:
        """Currently active control rate in Hz."""
        return self._rate

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
        """Signal a new episode and return the first observation snapshot.

        Blocks until at least one observation has been received for every
        expected key (joint state, camera frames, controller readings).

        Call once at the start of each episode.
        """
        self._check_connected()

        # Signal r2d2: start a new episode (for recording).
        self._transport.send_json(ResetEpisode())

        # Drain messages until all expected keys have data.
        deadline = time.monotonic() + _DISCOVERY_TIMEOUT
        while True:
            msg = self._transport.recv(timeout=_DISCOVERY_TIMEOUT)
            self._ingest(msg)

            snapshot = self._buffer.get()
            missing = set(self._obs_keys) - set(snapshot)
            if not missing:
                return snapshot

            if time.monotonic() > deadline:
                raise RobotDisconnectedError(
                    f"Timeout waiting for initial observations. Missing: {missing}"
                )

    def step(
        self,
        action: dict[str, np.ndarray] | None = None,
        *,
        reward: float = 0.0,
        done: bool = False,
    ) -> dict[str, np.ndarray]:
        """Send *action* to the robot, then return the latest observation.

        Blocks until the next observation arrives from r2d2.  Camera
        frames received while waiting are buffered and included in the
        returned snapshot.

        Parameters
        ----------
        action:
            Dictionary mapping action keys to NumPy arrays.  Pass
            ``None`` for a read‑only step (no command sent).
        reward:
            Optional scalar reward stored as dataset metadata.
        done:
            Optional terminal flag stored as dataset metadata.

        Returns
        -------
        dict
            Latest observation snapshot.
        """
        self._check_connected()

        # -- send action -----------------------------------------------------
        if action is not None:
            self._validate_action(action)
            self._transport.send_json(
                Action(
                    data={
                        k: v.astype(np.float64).ravel().tolist()
                        for k, v in action.items()
                    },
                    reward=reward,
                    done=done,
                )
            )

        # -- drain messages until the next Observation -----------------------
        while True:
            msg = self._transport.recv(timeout=5.0)
            self._ingest(msg)
            if isinstance(msg, Observation):
                return self._buffer.get()

    def close(self) -> None:
        """Disconnect from the station and release resources."""
        if self._closed:
            return
        self._closed = True
        if self._recording:
            try:
                self._transport.send_json(StopRecording())
            except Exception:
                pass
        try:
            self._transport.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Recording commands (thin wrappers — recording engine is on r2d2)
    # ------------------------------------------------------------------

    def start_recording(self, name: str) -> None:
        """Tell r2d2 to begin recording a LeRobot v3.0 dataset."""
        self._check_connected()
        self._transport.send_json(StartRecording(name=name))
        self._recording = True

    def stop_recording(self) -> None:
        """Tell r2d2 to finalize the current dataset."""
        self._check_connected()
        self._transport.send_json(StopRecording())
        self._recording = False

    def cancel_episode(self) -> None:
        """Tell r2d2 to discard the current in‑progress episode."""
        self._check_connected()
        self._transport.send_json(CancelEpisode())

    def recording(self, name: str) -> Recording:
        """Return a :class:`Recording` context manager for *name*.

        Usage::

            with robot.recording("session_001") as rec:
                while not rec.stop:
                    obs = robot.reset()
                    while not rec.stop and not rec.done:
                        obs = robot.step(policy(obs))
                    if rec.rerecord:
                        robot.cancel_episode()
                        rec.clear()
        """
        from c3po._recording import Recording

        return Recording(self, name)

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
    # Internal
    # ------------------------------------------------------------------

    def _check_connected(self) -> None:
        if self._closed:
            raise RobotDisconnectedError("Robot is closed.")
        if not self._transport.is_connected:
            raise RobotDisconnectedError("Connection to r2d2 was lost.")

    def _ingest(self, msg: Any) -> None:
        """Update the buffer from an incoming protocol message."""
        if isinstance(msg, Observation):
            for key, value in msg.data.items():
                self._buffer.update(key, np.array(value, dtype=np.float64))
        elif isinstance(msg, BinaryFrame):
            self._buffer.update_from_frame(msg.camera_name, msg.jpeg_bytes)

    def _validate_action(self, action: dict[str, np.ndarray]) -> None:
        """Check that *action* contains all required keys."""
        missing = set(self.action_keys) - set(action)
        if missing:
            raise ValueError(
                f"Action dict missing required keys: {missing}. "
                f"Expected keys: {self.action_keys}"
            )
