# c3po

## The Human‑Facing Protocol Droid

---

## 1. Purpose

c3po is the Python client library that researchers use to interact with any N‑Droids robot station. It provides a minimal, MDP‑inspired API through a single `Robot` class. It communicates with the station over WebSocket — no ROS 2 installation, no launch files, no driver configuration. The researcher writes a policy loop; c3po handles everything else.

---

## 2. Installation

```bash
pip install c3po
```

That’s it. Dependencies: `numpy` and `websocket-client`. No ROS 2, no system packages, no Docker. Works on Linux, macOS, and Windows.

---

## 3. API

### `Robot`

```python
from c3po import Robot

robot = Robot()
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rate` | `float` or `None` | `None` | Desired step rate in Hz. If `None`, uses the station’s native `control_rate`. |
| `host` | `str` | `"10.42.0.1"` | r2d2 server address. |
| `port` | `int` | `9090` | rosbridge WebSocket port. |

**Methods:**

- `reset() -> dict` — Returns the current observation snapshot. Resets the step‑rate limiter. Call once at the start of each episode.
- `step(action: dict | None = None) -> dict` — Sends the action, waits for the next observation (paced at the configured rate), and returns it. Pass `action=None` for a read‑only step (passive observation, no command sent).
- `close() -> None` — Disconnects and cleans up. Called automatically on context exit.

**Properties (populated from the `/describe` manifest):**

| Property | Type | Description |
|---|---|---|
| `station_model` | `str` | Station identifier (e.g., `"so101_teleop"`, `"aloha"`). |
| `control_rate` | `float` | Native control rate in Hz. |
| `arms` | `list[str]` | Arm names. Empty for stations with no actuated arms. |
| `cameras` | `list[str]` | Camera key names. |
| `controllers` | `list[str]` | Controller key names. |
| `action_keys` | `list[str]` | Keys that `step()` expects in the action dict. |
| `joint_names` | `dict[str, list[str]]` | Mapping from arm key to ordered joint name list. |

**Context manager:**

```python
with Robot() as robot:
    obs = robot.reset()
    while True:
        obs = robot.step(policy(obs))
# Robot is automatically closed on exit.
```

---

## 4. Observation and Action Format

### Observations

Observations are flat dictionaries of NumPy arrays. Every key and its shape is determined by the station’s `/describe` manifest. The researcher introspects `robot.arms`, `robot.cameras`, and `robot.controllers` to know what keys are available.

**Single‑arm station with a leader controller and two cameras:**

```python
obs = {
    "joint_position": np.ndarray(dtype=float64, shape=(6,)),
    "joint_velocity": np.ndarray(dtype=float64, shape=(6,)),
    "leader/joint_position": np.ndarray(dtype=float64, shape=(6,)),
    "front_rgb": np.ndarray(dtype=uint8, shape=(720, 1280, 3)),
    "follower/wrist_rgb": np.ndarray(dtype=uint8, shape=(480, 640, 3)),
}
```

**Bimanual ALOHA station:**

```python
obs = {
    "left_follower/joint_position":  np.ndarray(dtype=float64, shape=(6,)),
    "right_follower/joint_position": np.ndarray(dtype=float64, shape=(6,)),
    "left_leader/joint_position":    np.ndarray(dtype=float64, shape=(6,)),
    "right_leader/joint_position":   np.ndarray(dtype=float64, shape=(6,)),
    "left_follower/wrist_rgb":       np.ndarray(dtype=uint8, shape=(480, 640, 3)),
    "right_follower/wrist_rgb":      np.ndarray(dtype=uint8, shape=(480, 640, 3)),
    "front_rgb":                     np.ndarray(dtype=uint8, shape=(720, 1280, 3)),
}
```

### Actions

Actions are flat dictionaries matching the keys listed in `robot.action_keys`:

```python
robot.action_keys  # → ["follower/joint_position"]

action = {
    "follower/joint_position": np.ndarray(dtype=float64, shape=(6,)),
}
```

### Key naming conventions

| Pattern | Meaning |
|---|---|
| `{arm}/joint_position` | Joint angles (radians) for the named arm. Omitted for single‑arm stations. |
| `{arm}/joint_velocity` | Joint velocities (rad/s) for the named arm. |
| `{camera}` | Camera image. Camera names encode arm affiliation via `/` when applicable. |
| `{controller}/state` | Controller reading. Dtype and shape depend on controller type. |

---

## 5. Step Timing

`step()` manages the control loop cadence automatically. It sends the action immediately, then sleeps for whatever time remains in the current cycle to maintain a precise `rate` Hz rhythm.

```
Policy rate: 50 Hz (20 ms per step)
Camera 1:    30 Hz (33 ms per frame)
Camera 2:    60 Hz (17 ms per frame)

Time: 0ms    20ms   40ms   60ms   80ms   100ms
      |      |      |      |      |      |
Cam1: F0------------F1------------F2-------
Cam2: F0-----F1-----F2-----F3-----F4------

step() at 0ms:  returns latest: {cam1: F0, cam2: F1}
step() at 20ms: returns latest: {cam1: F0, cam2: F2}   ← cam1 reused (still latest)
step() at 40ms: returns latest: {cam1: F1, cam2: F3}
```

Each `step()` returns the most recent data from every sensor at call time — it never blocks waiting for a slow sensor to produce new data. Camera frames may be reused across steps when the camera rate is slower than the policy rate. This is correct behaviour: reusing a 33 ms‑old frame is better than delaying the control loop.

If `step()` is called before the previous cycle’s period has elapsed, it sleeps for the remainder. If the policy takes longer than one period, it returns immediately with no sleep. The rate limiter tracks wall clock, so a single slow step does not cascade.

---

## 6. Usage Examples

### Single‑arm policy loop

```python
from c3po import Robot

with Robot() as robot:
    print(f"Station: {robot.station_model}")
    print(f"Arms: {robot.arms}")
    print(f"Action keys: {robot.action_keys}")

    obs = robot.reset()
    while True:
        action = policy(obs)
        obs = robot.step(action)
```

### Leader‑follower teleoperation

```python
from c3po import Robot

with Robot() as robot:
    obs = robot.reset()
    while True:
        # The leader arm's joint position is in the observation dict.
        # Map it directly to the follower's action key.
        obs = robot.step(
            {"follower/joint_position": obs["leader/joint_position"]}
        )
```

### Bimanual teleoperation

```python
from c3po import Robot

with Robot() as robot:
    obs = robot.reset()
    while True:
        obs = robot.step({
            "left_follower/joint_position":  obs["left_leader/joint_position"],
            "right_follower/joint_position": obs["right_leader/joint_position"],
        })
```

### Recording a session

```python
from c3po import Robot, Recorder

with Robot() as robot, Recorder(robot, "demonstrations.h5") as rec:
    obs = robot.reset()
    rec.start_episode()

    while not done:
        action = policy(obs)
        obs = robot.step(action)
        reward = compute_reward(obs)
        done = check_done(obs)
        rec.record(obs, action, reward, done)

    rec.end_episode()
```

### Replaying recorded data

```python
from c3po import ReplayClient

with ReplayClient("demonstrations.h5") as robot:
    for obs, action, reward, done in robot.episodes():
        # Offline policy evaluation or training
        predicted = policy(obs)
```

### Introspecting an unknown station

```python
from c3po import Robot

with Robot() as robot:
    print(f"Connected to: {robot.station_model}")
    print(f"Control rate: {robot.control_rate} Hz")
    print(f"Arms:          {robot.arms}")
    print(f"Cameras:       {robot.cameras}")
    print(f"Controllers:   {robot.controllers}")
    print(f"Action keys:   {robot.action_keys}")
```

---

## 7. Recording and Replay

### Recorder

`Recorder` is a context manager that writes observations, actions, rewards, and terminal flags to an HDF5 file. It embeds the station manifest in the file metadata so replay clients can reconstruct the exact observation structure.

```python
with Recorder(robot, "session.h5") as rec:
    obs = robot.reset()
    rec.start_episode()
    while not done:
        action = policy(obs)
        obs = robot.step(action)
        rec.step()  # records current (obs, action, next_obs, reward, done)
    rec.end_episode()
```

### ReplayClient

`ReplayClient` wraps a recorded HDF5 file and exposes the same `Robot` API. `reset()` seeks to the first frame of the first episode. `step()` advances one frame. The `episodes()` generator yields `(obs, action, reward, done)` tuples for offline iteration.

### LeRobot compatibility

A `to_lerobot_format()` utility converts recorded observations to the LeRobot dataset format:

```python
from c3po import to_lerobot_format

lerobot_obs = to_lerobot_format(obs)
# → {"observation.state": obs["joint_position"],
#    "observation.images.front_rgb": obs["front_rgb"], ...}
```

This allows data collected with c3po to be used with LeRobot’s training pipelines without format friction.

---

## 8. Error Handling

| Exception | Raised when |
|---|---|
| `RobotNotFoundError` | No `/describe` service responds within the timeout (Ethernet cable unplugged, r2d2 not running). |
| `RobotDisconnectedError` | The WebSocket connection drops mid‑session. |
| `ValueError` | Action dict is missing required keys or has wrong array shapes (validated against the manifest). |
| `IncompatibleVersionError` | r2d2 reports an API version that c3po does not support. |

All exceptions are subclasses of `RuntimeError`. They include descriptive messages to help diagnose the problem (e.g., “Action dict missing key ‘left_follower/joint_position’. Expected keys: [...]”).

---

## 9. Internal Architecture

```
c3po/src/c3po/
├── __init__.py           # Exports Robot, Recorder, ReplayClient, exceptions, to_lerobot_format
├── robot.py              # Public Robot class
├── _transport/
│   ├── __init__.py
│   ├── _base.py          # Transport ABC
│   └── _websocket.py     # WebSocket transport via rosbridge protocol
├── _manifest.py          # /describe service client, manifest parsing
├── _buffer.py            # Thread‑safe latest‑value observation buffer
├── _recorder.py          # Recorder and ReplayClient
├── _lerobot.py           # LeRobot format conversion
└── exceptions.py         # Custom exception classes
```

### Connection lifecycle

1. `Robot.__init__` opens a WebSocket to `ws://10.42.0.1:9090`.
2. It calls the `/describe` service over rosbridge (blocks with timeout).
3. The manifest is parsed and stored. Properties (`arms`, `cameras`, etc.) are populated.
4. Subscriptions are created for every observation topic listed in the manifest.
5. A publisher is created for every action key.
6. A background thread reads incoming WebSocket messages and updates the observation buffer.
7. `reset()` and `step()` read from the buffer.

### Observation buffer

The buffer stores the latest value received for each observation key. `step()` performs a non‑blocking read — it returns whatever is the most recent value for each key at call time. No timestamp alignment is needed because the latest available frame from each sensor is the best data the policy can act on at that instant.

---

## 10. Versioning and Compatibility

c3po follows semantic versioning. It is compatible with any r2d2 server that reports the same `api_version` major number in its `/describe` manifest. If the client detects an incompatible API version, it raises `IncompatibleVersionError` with a clear upgrade message.

The `/describe` manifest is the sole compatibility contract. As long as the manifest structure, topic naming, and rosbridge protocol remain backward‑compatible within a major version, c3po and r2d2 can be upgraded independently.
