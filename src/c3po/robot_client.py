"""Public API for the c3po client library.

Provides :class:`RobotClient` — a single-object entry point that discovers
an N‑Droids robot station and exposes a simple ``get_observation()`` /
``send_action()`` interface.

Usage::

    from c3po import RobotClient

    with RobotClient() as robot:
        while True:
            obs = robot.get_observation()
            action = policy(obs)
            robot.send_action(action)
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# rclpy imports — provided by the system ROS 2 installation, not by pip
# ---------------------------------------------------------------------------
import rclpy
from c3po_msgs.msg import Action
from c3po_msgs.srv import Describe
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from c3po._buffer import _ObservationBuffer
from c3po.exceptions import RobotDisconnectedError, RobotNotFoundError

# ---------------------------------------------------------------------------
# Internal: rclpy node singleton
# ---------------------------------------------------------------------------

_node_lock: threading.Lock = threading.Lock()
_node: Node | None = None
_executor: SingleThreadedExecutor | None = None


def _ensure_node() -> Node:
    """Return the process-global rclpy node, creating it on first call.

    Does **not** start the background executor — that happens after
    discovery completes (see :func:`_start_executor`).
    """
    global _node
    with _node_lock:
        if _node is not None:
            return _node
        rclpy.init()
        _node = Node("c3po_client")
        return _node


def _start_executor(node: Node) -> None:
    """Start the background executor thread for *node*."""
    global _executor
    with _node_lock:
        if _executor is not None:
            return
        _executor = SingleThreadedExecutor()
        _executor.add_node(node)
        thread = threading.Thread(target=_executor.spin, daemon=True, name="c3po-exec")
        thread.start()


def _shutdown_node() -> None:
    """Shut down the global rclpy node if it was created."""
    global _node, _executor
    with _node_lock:
        if _executor is not None:
            _executor.shutdown()
            _executor = None
        if _node is not None:
            _node.destroy_node()
            _node = None
        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DISCOVERY_TIMEOUT = 10.0  # seconds
_OBS_TIMEOUT = 5.0  # seconds (5× the expected period)


class RobotClient:
    """Client for an N‑Droids robot station.

    Discovers the station via the ``/describe`` ROS 2 service and provides
    a minimal MDP‑inspired API.
    """

    def __init__(self, station_name: str | None = None) -> None:
        # -- create (or reuse) the singleton node ---------------------------------
        node = _ensure_node()

        # -- discovery (synchronous, before the background executor starts) --------
        try:
            cli = node.create_client(Describe, "/describe")
            if not cli.wait_for_service(timeout_sec=_DISCOVERY_TIMEOUT):
                raise RobotNotFoundError(
                    f"No /describe service found within {_DISCOVERY_TIMEOUT} s. "
                    "Is the Ethernet cable connected?"
                )

            req = Describe.Request()
            future = cli.call_async(req)
            rclpy.spin_until_future_complete(
                node, future, timeout_sec=_DISCOVERY_TIMEOUT
            )
            if not future.done() or future.result() is None:
                raise RobotNotFoundError("Failed to call /describe service.")

            response: Describe.Response = future.result()
            self._manifest: dict[str, Any] = {
                "robot_model": response.robot_model,
                "joint_count": response.joint_count,
                "control_rate": response.control_rate,
                "camera_names": list(response.camera_names),
                "command_mode": response.command_mode,
                "joint_names": list(response.joint_names),
            }
        except RobotNotFoundError:
            _shutdown_node()
            raise
        except Exception:
            _shutdown_node()
            raise

        # -- start the background executor (subscriptions / actions) --------------
        _start_executor(node)
        self._node = node

        # -- observation subscribers ----------------------------------------------
        self._buffer = _ObservationBuffer()
        self._obs_keys: list[str] = []

        # Joint state (always present)
        self._joint_sub = node.create_subscription(
            JointState,
            "/observations/joint_position",
            self._on_joint_state,
            10,
        )
        self._obs_keys.extend(["joint_position", "joint_velocity"])

        # Camera subscribers (added when camera support lands)
        for cam_name in self._manifest.get("camera_names", []):
            pass  # placeholder

        self._buffer.configure(self._obs_keys)

        # -- command publisher ----------------------------------------------------
        self._cmd_pub = node.create_publisher(
            Action,
            f"/commands/{self._manifest['command_mode']}",
            10,
        )

        self._closed = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def robot_model(self) -> str:
        return self._manifest["robot_model"]

    @property
    def joint_count(self) -> int:
        return self._manifest["joint_count"]

    @property
    def control_rate(self) -> float:
        return self._manifest["control_rate"]

    @property
    def camera_names(self) -> list[str]:
        return self._manifest.get("camera_names", [])

    # ------------------------------------------------------------------
    # Observation / action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, np.ndarray]:
        """Block until a fresh observation bundle arrives.

        Returns a dictionary of NumPy arrays.  Keys include
        ``joint_position``, ``joint_velocity``, and one key per camera
        (e.g. ``front_rgb``).
        """
        if self._closed:
            raise RobotDisconnectedError("RobotClient is closed.")
        return self._buffer.get(timeout=_OBS_TIMEOUT)

    def send_action(self, action: dict[str, np.ndarray]) -> None:
        """Publish an action to the robot.

        *action* must contain the key matching the station's ``command_mode``
        (e.g. ``"joint_position"`` for a position-controlled robot).
        """
        if self._closed:
            raise RobotDisconnectedError("RobotClient is closed.")

        mode = self._manifest["command_mode"]
        if mode not in action:
            raise ValueError(
                f"Action dict must contain key '{mode}'. "
                f"Got keys: {list(action.keys())}"
            )

        msg = Action()
        msg.mode = mode
        msg.joint_names = self._manifest.get("joint_names", [])
        msg.data = action[mode].astype(np.float64).ravel().tolist()
        self._cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Cleanly shut down the ROS 2 connection."""
        if self._closed:
            return
        self._closed = True
        _shutdown_node()

    def __enter__(self) -> RobotClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal callbacks (run in executor thread)
    # ------------------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        self._buffer.update("joint_position", np.array(msg.position, dtype=np.float64))
        self._buffer.update("joint_velocity", np.array(msg.velocity, dtype=np.float64))
