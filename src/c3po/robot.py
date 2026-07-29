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
    SpecRequest,
    SpecResponse,
    StartRecording,
    StatusMessage,
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
        self._on_status_callback = None
        self._dataset_urls: dict[str, str] = {}  # name -> HTTP URL
        self._host = host

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

    @property
    def on_status(self):
        """Callback invoked on StatusMessage from r2d2.

        Set to a callable ``f(event, message, data)`` to receive
        asynchronous lifecycle events (episode finalized, watchdog
        triggered, etc.).  Set to ``None`` to disable.
        """
        return self._on_status_callback

    @on_status.setter
    def on_status(self, callback):
        self._on_status_callback = callback

    @property
    def camera_stream_info(self) -> dict[str, dict[str, list[int]]]:
        """Mapping from camera name to capture and streaming resolutions.

        Each entry is a dict with ``"capture_resolution"`` (what the
        hardware records) and ``"stream_resolution"`` (what arrives
        over the WebSocket — may be lower if a streaming cap is active).
        """
        info: dict[str, dict[str, list[int]]] = {}
        for cam in self._manifest.get("cameras", []):
            name = cam["name"]
            capture = cam.get("resolution", [0, 0])
            stream = cam.get("stream_resolution", capture)
            info[name] = {
                "capture_resolution": list(capture),
                "stream_resolution": list(stream),
            }
        return info

    @property
    def controller_capabilities(self) -> dict[str, list[str]]:
        """Mapping from controller name to its capabilities list.

        Capability strings include ``"auto_reset"`` and
        ``"haptic_feedback"``.  An empty list means a passive,
        read-only controller.
        """
        return {
            c["name"]: c.get("capabilities", [])
            for c in self._manifest.get("controllers", [])
        }

    @property
    def leader_home_position(self) -> dict[str, list[float] | None]:
        """Mapping from controller name to its home position.

        Returns ``None`` for controllers without a configured home
        position.  Only meaningful when ``"auto_reset"`` is in the
        controller's capabilities.
        """
        return {
            c["name"]: c.get("home_position")
            for c in self._manifest.get("controllers", [])
        }

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

    def download_dataset(
        self, name: str, dest: str | Path = ".", *, show_progress: bool = True
    ) -> Path:
        """Download a recorded dataset from r2d2's HTTP server.

        Pulls the complete LeRobot dataset tree (``data/``, ``meta/``,
        ``videos/``) over the point-to-point Ethernet link.  Uses the
        HTTP URL published by r2d2's ``dataset_ready`` notification.

        Parameters
        ----------
        name:
            Dataset name (same as passed to :meth:`start_recording`).
        dest:
            Local directory to save into.  A subdirectory *name* is
            created inside *dest*.
        show_progress:
            If ``True`` (default), log download progress.

        Returns
        -------
        Path
            The local dataset directory (``dest / name``).

        Raises
        ------
        ValueError
            If *name* has no known URL (e.g. not yet recorded or
            ``dataset_ready`` notification was missed).
        RuntimeError
            If the HTTP server is unreachable.
        """
        from html.parser import HTMLParser
        from pathlib import Path
        from urllib.error import URLError
        from urllib.request import urlopen, urlretrieve

        # Resolve the dataset URL.
        url_base = self._dataset_urls.get(name)
        if url_base is None:
            # Fallback: construct from host + default HTTP port.
            url_base = f"http://{self._host}:9091/{name}/"
        url_base = url_base.rstrip("/") + "/"

        dest_dir = Path(dest) / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # -- HTML link extractor -------------------------------------------------
        class _LinkParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.links: list[str] = []

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag == "a":
                    for attr, val in attrs:
                        if attr == "href" and val:
                            self.links.append(val)

        # -- recursive download --------------------------------------------------
        def _download_tree(remote_dir: str, local_dir: Path, depth: int = 0) -> None:
            local_dir.mkdir(parents=True, exist_ok=True)
            try:
                with urlopen(remote_dir, timeout=10.0) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
            except URLError as e:
                if "Connection refused" in str(e) or "Errno 111" in str(e):
                    raise RuntimeError(
                        f"Failed to reach r2d2 HTTP server at {remote_dir}: {e}\n"
                        "Is port 9091 exposed?  Make sure the Docker container "
                        "was started with '-p 9091:9091' and has been rebuilt "
                        "after Phase 12 (dataset forwarding)."
                    )
                raise RuntimeError(
                    f"Failed to reach r2d2 HTTP server at {remote_dir}: {e}"
                )

            parser = _LinkParser()
            parser.feed(html)

            for link in parser.links:
                if link.startswith("?") or link == "/" or link.startswith(".."):
                    continue
                remote_url = remote_dir.rstrip("/") + "/" + link
                local_path = local_dir / link.rstrip("/")

                if link.endswith("/"):
                    # Directory: recurse.
                    _download_tree(remote_url + "/", local_path, depth + 1)
                else:
                    # File: download.
                    if show_progress:
                        logger.info("  downloading %s", local_path)
                    try:
                        urlretrieve(remote_url, str(local_path))
                    except Exception as e:
                        logger.warning("Failed to download %s: %s", remote_url, e)

        logger.info("Downloading dataset '%s' from %s ...", name, url_base)
        _download_tree(url_base, dest_dir)

        # -- verify --------------------------------------------------------------
        meta = dest_dir / "meta" / "info.json"
        if not meta.exists():
            logger.warning(
                "Dataset '%s' downloaded but meta/info.json not found — "
                "directory may be incomplete.",
                name,
            )
        else:
            total_size = sum(
                f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()
            )
            logger.info(
                "Dataset '%s' downloaded to %s (%.1f MB)",
                name,
                dest_dir,
                total_size / (1024 * 1024),
            )
        return dest_dir

    def spec(self) -> SpecResponse:
        """Request live runtime metrics from r2d2.

        Returns a :class:`SpecResponse` with effective control rate,
        camera stats, recording state, and watchdog trigger count.
        Use ``str(robot.spec())`` for a human-readable display.
        """
        self._check_connected()
        self._transport.send_json(SpecRequest())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            msg = self._transport.recv(timeout=min(1.0, deadline - time.monotonic()))
            if isinstance(msg, SpecResponse):
                return msg
            # Drain Observations, BinaryFrames, and StatusMessages.
        raise RuntimeError("Timed out waiting for SpecResponse from r2d2")

    def recording(self, name: str) -> "Recording":
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
        if hasattr(self, "_closed") and not self._closed:
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
            self._buffer.update_from_frame(msg)
        elif isinstance(msg, StatusMessage):
            # Capture dataset_ready URLs for later download.
            if msg.event == "dataset_ready" and "url" in (msg.data or {}):
                name = msg.data.get("name", "")
                if name:
                    self._dataset_urls[name] = msg.data["url"]
            level = (
                logging.WARNING
                if msg.event == "watchdog_triggered"
                else logging.INFO
            )
            logger.log(
                level,
                "[r2d2] %s: %s %s",
                msg.event,
                msg.message,
                msg.data if msg.data else "",
            )
            if self._on_status_callback is not None:
                try:
                    self._on_status_callback(msg.event, msg.message, msg.data)
                except Exception:
                    logger.warning(
                        "on_status callback raised an exception", exc_info=True
                    )

    def _validate_action(self, action: dict[str, np.ndarray]) -> None:
        """Check that *action* contains all required keys."""
        missing = set(self.action_keys) - set(action)
        if missing:
            raise ValueError(
                f"Action dict missing required keys: {missing}. "
                f"Expected keys: {self.action_keys}"
            )
