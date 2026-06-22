
# c3po Technical Specification

## The Human‑Facing Protocol Droid

---

## 1. Purpose

c3po is the Python client library used by AI researchers to interact with any N‑Droids‑enabled robot station. It provides a minimal, MDP‑inspired API that hides all ROS 2 machinery behind a simple `RobotClient` class. It is installed via `pip` and works within the researcher’s own project environment.

---

## 2. Installation & Prerequisites

**Prerequisite:** ROS 2 Humble (or later) must be installed on the researcher’s machine. This is a one‑time, lab‑wide setup:

```bash
sudo apt update && sudo apt install ros-humble-ros-base
```

The c3po package is then installed with:

```bash
pip install c3po
```

Dependencies: `numpy`, and the system‑provided `rclpy` (imported at runtime; no custom wheels are required). The package is pure Python and platform‑independent as long as ROS 2 is present.

---

## 3. API Design

### `RobotClient`

```python
from c3po import RobotClient

robot = RobotClient(station_name=None)
```

**Constructor:**
- `station_name` (optional): If provided, discovers only the robot with a matching ROS 2 node name or mDNS hostname. If `None`, connects to the first `/describe` service found.

**Methods:**

- `get_observation() -> dict`  
  Blocks until a complete, synchronised set of observations (joint state + all camera frames with matching timestamps) is available. Returns a dictionary:
  ```python
  {
      "joint_position": np.ndarray(dtype=float32),
      "joint_velocity": np.ndarray(dtype=float32),
      "<camera_name>_rgb": np.ndarray(dtype=uint8, shape=(H,W,3)),
      # ... optional depth, etc.
  }
  ```
  The exact keys depend on the robot’s manifest (retrieved via `/describe`). The method raises `RobotDisconnectedError` if the ROS 2 connection is lost.

- `send_action(action: dict) -> None`  
  Accepts a dictionary with keys matching the command mode (e.g., `{"joint_position": np.array(...)}`). Publishes the corresponding ROS 2 message immediately. Raises `ValueError` if the action format does not match the manifest.

- `close()`  
  Shuts down the ROS 2 node cleanly. Called automatically on context exit (`with RobotClient() as robot:`).

**Properties:**
- `robot_model`, `joint_count`, `control_rate`, `camera_names` — populated from the manifest.

---

## 4. Internal Architecture

```
c3po/
├── __init__.py           # Exposes RobotClient
├── robot_client.py       # Public API, lazy ROS 2 init
├── _node_manager.py      # Singleton rclpy node, background executor
├── _discovery.py         # Service client for /describe
├── _subscriber.py        # Dynamically creates subscribers for each observation topic
├── _publisher.py         # Dynamic publisher for command topics
├── _sync.py              # Observation buffer and timestamp alignment
└── exceptions.py         # Custom exceptions
```

**Initialisation flow:**
1. On first `RobotClient` creation, `_node_manager` initialises `rclpy` (if not already) and starts a background executor thread.
2. A ROS 2 node named `c3po_client` is created.
3. `_discovery` calls the `/describe` service (blocking with timeout). The returned manifest is stored.
4. `_subscriber` creates a `Subscription` for each listed observation topic.
5. `_publisher` creates a `Publisher` for the command topic corresponding to the `command_mode`.
6. `_sync` collects incoming messages and assembles a time‑synchronised observation bundle using the closest‑timestamp heuristic.

**Observation synchronisation:**  
The `get_observation()` call waits until at least one new message has arrived on each subscribed topic since the last call. It then returns the latest values, aligned by header stamp. This guarantees that the researcher always gets a self‑consistent state.

**Thread safety:**  
The background executor and the user’s loop run on separate threads. A thread‑safe queue is used to pass actions to the publisher.

---

## 5. Example Usage

```python
from c3po import RobotClient
from my_policy import load_policy

policy = load_policy("checkpoint.pt")

with RobotClient() as robot:
    print(f"Connected to {robot.robot_model} at {robot.control_rate} Hz")
    while True:
        obs = robot.get_observation()
        action = policy(obs)
        robot.send_action(action)
```

The researcher runs this script with `python run.py`. No launch files, no ROS 2 CLI commands.

---

## 6. Error Handling

- **Discovery timeout:** If no `/describe` service appears within 10 seconds, `RobotNotFoundError` is raised (the Ethernet cable might be unplugged).
- **Command format mismatch:** If the action dict does not contain the expected key or array shape, `ValueError` is raised immediately.
- **Connection loss:** If any subscription stops receiving messages for longer than 5× the expected period, `RobotDisconnectedError` is raised inside `get_observation()`.

---

## 7. Versioning & Compatibility

c3po follows semantic versioning. It is designed to be compatible with any r2d2 server that implements the same `N‑Droids API version`. The API version number is included in the `/describe` response. If the client detects an incompatible API version, it raises a clear error with upgrade instructions.

---

## 8. Distribution

c3po is published to the lab’s internal PyPI index. The source repository includes CI that runs integration tests against a mock r2d2 container to ensure the ROS 2 contract is upheld.
