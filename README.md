# c3po

## The Human‑Facing Protocol Droid

---

## 1. Purpose

c3po is the Python client library that researchers use to interact with any N‑Droids robot station. It provides a minimal, MDP‑inspired API through a single `Robot` class, communicating with the station over a lightweight WebSocket protocol. No ROS 2, no launch files, no driver configuration. The researcher writes a policy loop; c3po handles everything else.

---

## 2. Installation

```bash
pip install c3po
```

That's it. Dependencies: `numpy`, `Pillow`, and `websocket-client`. No PyTorch, no CUDA, no ROS 2, no Docker. Works on Linux, macOS, and Windows.

---

## 3. API

### `Robot`

```python
from c3po import Robot

with Robot() as robot:
    obs = robot.reset()
    while True:
        obs = robot.step(policy(obs))
```

**Constructor parameters:**

| Parameter | Type  | Default       | Description          |
| --------- | ----- | ------------- | -------------------- |
| `host`    | `str` | `"10.42.0.1"` | r2d2 server address. |
| `port`    | `int` | `9090`        | WebSocket port.      |

The control rate is determined by r2d2 — no `rate` parameter needed. c3po's `step()` blocks until the next observation arrives, naturally pacing the loop at the station's native rate.

**Methods:**

| Method                                                 | Description                                                                                                                                                                                                                 |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reset() -> dict`                                      | Sends a new‑episode signal to r2d2. Returns the current observation snapshot. Blocks until all expected keys have data. Call at the start of each episode.                                                                  |
| `step(action=None, *, reward=0.0, done=False) -> dict` | Sends the action (if provided) to r2d2, blocks until the next observation arrives, and returns it. `reward` and `done` are optional metadata stored with the frame in the dataset. Pass `action=None` for a read‑only step. |
| `close() -> None`                                      | Disconnects and releases resources. Called automatically on context manager exit.                                                                                                                                           |
| `start_recording(name: str) -> None`                   | Tells r2d2 to begin recording a LeRobot v3.0 dataset. Episode boundaries are signaled by `reset()` calls.                                                                                                                   |
| `stop_recording() -> None`                             | Tells r2d2 to finalize and close the current dataset.                                                                                                                                                                       |
| `cancel_episode() -> None`                             | Tells r2d2 to discard the current in‑progress episode.                                                                                                                                                                      |
| `recording(name: str) -> RecordingSession`             | Context manager combining `start_recording` / `stop_recording` with built‑in keyboard controls. Returns a `RecordingSession` with `stop`, `done`, and `rerecord` flags. See section 6.                                      |

**Properties (populated from the `describe` manifest):**

| Property        | Type                   | Description                                                             |
| --------------- | ---------------------- | ----------------------------------------------------------------------- |
| `station_model` | `str`                  | e.g. `"so101_teleop"`, `"aloha"`.                                       |
| `control_rate`  | `float`                | Station's native control rate in Hz.                                    |
| `arms`          | `list[str]`            | Arm names (e.g. `["follower"]`, `["left_follower", "right_follower"]`). |
| `cameras`       | `list[str]`            | Camera key names present in observations.                               |
| `controllers`   | `list[str]`            | Controller key names present in observations.                           |
| `action_keys`   | `list[str]`            | Keys that `step()` expects in the action dictionary.                    |
| `joint_names`   | `dict[str, list[str]]` | Mapping from arm key to ordered joint name list.                        |

---

## 4. Observation and Action Format

### Observations

Flat dictionaries of NumPy arrays. Every key and shape is determined by the station's `describe` manifest.

**Single‑arm station with a leader and two cameras:**

```python
obs = {
    "follower/joint_position": np.ndarray(dtype=float64, shape=(6,)),
    "follower/joint_velocity": np.ndarray(dtype=float64, shape=(6,)),
    "leader/joint_position":  np.ndarray(dtype=float64, shape=(6,)),
    "front_rgb":              np.ndarray(dtype=uint8, shape=(720, 1280, 3)),
    "wrist_rgb":              np.ndarray(dtype=uint8, shape=(480, 640, 3)),
}
```

**Bimanual ALOHA station:**

```python
obs = {
    "left_follower/joint_position":  np.ndarray(dtype=float64, shape=(6,)),
    "right_follower/joint_position": np.ndarray(dtype=float64, shape=(6,)),
    "left_leader/joint_position":    np.ndarray(dtype=float64, shape=(6,)),
    "right_leader/joint_position":   np.ndarray(dtype=float64, shape=(6,)),
    "left_wrist_rgb":                np.ndarray(dtype=uint8, shape=(480, 640, 3)),
    "right_wrist_rgb":               np.ndarray(dtype=uint8, shape=(480, 640, 3)),
    "front_rgb":                     np.ndarray(dtype=uint8, shape=(720, 1280, 3)),
}
```

### Actions

Flat dictionaries matching `robot.action_keys`:

```python
# Single arm
{"follower/joint_position": np.ndarray(dtype=float64, shape=(6,))}

# Bimanual
{
    "left_follower/joint_position":  np.ndarray(dtype=float64, shape=(6,)),
    "right_follower/joint_position": np.ndarray(dtype=float64, shape=(6,)),
}
```

`step()` validates that all required keys are present and raises `ValueError` with a descriptive message on mismatch.

### Key naming conventions

| Pattern                       | Meaning                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| `{arm}/joint_position`        | Joint angles (radians).                                                   |
| `{arm}/joint_velocity`        | Joint velocities (rad/s).                                                 |
| `{camera}`                    | Camera image. Names with arm affiliation use `/` (e.g. `left/wrist_rgb`). |
| `{controller}/joint_position` | Controller reading.                                                       |

---

## 5. Step Timing

c3po does not track wall‑clock time. r2d2 owns the control loop cadence and streams observations at a precise rate. c3po's `step(action)` sends the action, then blocks until the next observation arrives. The researcher's loop is naturally paced at the station's control rate:

```
r2d2 cycle (50 Hz, 20 ms):
  read sensors ── send obs ── wait for action ── send to robot ── sleep remainder

c3po cycle:
  receive obs ── policy(obs) ── send action ── block on next obs
```

If the policy is fast (3 ms), `step()` blocks for the remaining 17 ms until the next observation. If the policy is slow (25 ms), r2d2 detects the delay and runs the next cycle immediately — no double‑waiting. Each `step()` returns the most recent data from every sensor without blocking on slow topics.

---

## 6. Recording

### Basic recording

```python
with Robot() as robot:
    robot.start_recording("session_001")
    obs = robot.reset()
    for _ in range(300):
        obs = robot.step(policy(obs))
    robot.stop_recording()
```

This produces a LeRobot v3.0 dataset on the NUC at `/datasets/session_001/`. The researcher copies it to their machine however they prefer (scp, rsync, NFS mount).

### Recording with keyboard controls

c3po includes a built‑in keyboard listener that captures three signals from the terminal:

| Key   | Signal     | Effect                                             |
| ----- | ---------- | -------------------------------------------------- |
| **n** | `done`     | End the current episode (save it, start the next). |
| **r** | `rerecord` | Discard the current episode and re‑record it.      |
| **q** | `stop`     | Stop recording and finalize the dataset.           |

These are the same signals LeRobot uses, with the same key bindings (LeRobot uses n/r/q as the letter alternatives to arrow keys). No extra dependencies — the listener uses only Python stdlib (`termios`, `tty`, `select`) on Unix. On Windows (where `termios` is unavailable) the listener gracefully degrades to a no‑op and the researcher uses Ctrl+C.

```python
from c3po import Robot

with Robot() as robot:
    with robot.recording("session_001") as rec:
        while not rec.stop:
            obs = robot.reset()
            while not rec.stop and not rec.done:
                obs = robot.step(policy(obs))
            if rec.rerecord:
                robot.cancel_episode()
                rec.clear()          # reset done/rerecord flags for next iteration
```

`robot.recording()` is a convenience context manager that:

1. Sends `start_recording` on enter.
2. Starts the keyboard listener in a background thread.
3. Exposes `rec.stop`, `rec.done`, and `rec.rerecord` — boolean flags set by key presses.
4. Calls `rec.clear()` to reset `done` and `rerecord` (but not `stop`) between episodes.
5. Sends `stop_recording` and stops the listener on exit.

The researcher checks the flags at natural loop boundaries and calls `cancel_episode()` + `clear()` on re‑record. Ctrl+C still works for hard stops — the context manager's exit handler ensures `stop_recording` is sent before disconnecting.

### Recording with task‑aware episode termination

Task completion can coexist with keyboard controls:

```python
with Robot() as robot:
    with robot.recording("session_001") as rec:
        while not rec.stop:
            obs = robot.reset()
            while not rec.stop and not rec.done:
                if task_complete(obs):
                    obs = robot.step(policy(obs), reward=1.0, done=True)
                    break
                obs = robot.step(policy(obs), reward=0.0)
            if rec.rerecord:
                robot.cancel_episode()
                rec.clear()
```

The `done` kwarg on `step()` is dataset metadata — it marks the terminal frame in the parquet file. It does not affect the control loop. The keyboard flags (`rec.done`, `rec.stop`) control the loop itself.

---

## 7. Usage Examples

### Leader‑follower teleoperation

```python
from c3po import Robot

with Robot() as robot:
    with robot.recording("teleop_demo") as rec:
        while not rec.stop:
            obs = robot.reset()
            while not rec.stop and not rec.done:
                obs = robot.step(
                    {"follower/joint_position": obs["leader/joint_position"]}
                )
            if rec.rerecord:
                robot.cancel_episode()
                rec.clear()
```

### Bimanual teleoperation

```python
from c3po import Robot

with Robot() as robot:
    with robot.recording("bimanual_demo") as rec:
        while not rec.stop:
            obs = robot.reset()
            while not rec.stop and not rec.done:
                obs = robot.step({
                    "left_follower/joint_position":  obs["left_leader/joint_position"],
                    "right_follower/joint_position": obs["right_leader/joint_position"],
                })
            if rec.rerecord:
                robot.cancel_episode()
                rec.clear()
```

### Policy loop with rewards

```python
from c3po import Robot

with Robot() as robot:
    with robot.recording("policy_rollout") as rec:
        while not rec.stop:
            obs = robot.reset()
            while not rec.stop and not rec.done:
                action = policy(obs)
                obs = robot.step(action, reward=compute_reward(obs),
                                 done=check_done(obs))
            if rec.rerecord:
                robot.cancel_episode()
                rec.clear()
```

### Introspecting an unknown station

```python
from c3po import Robot

with Robot() as robot:
    print(f"Station:      {robot.station_model}")
    print(f"Control rate: {robot.control_rate} Hz")
    print(f"Arms:         {robot.arms}")
    print(f"Cameras:      {robot.cameras}")
    print(f"Controllers:  {robot.controllers}")
    print(f"Action keys:  {robot.action_keys}")
```

---

## 8. Error Handling

| Exception                  | Raised when                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `RobotNotFoundError`       | No `describe` response within the timeout (Ethernet unplugged, r2d2 not running, wrong IP). |
| `RobotDisconnectedError`   | WebSocket connection drops mid‑session.                                                     |
| `ValueError`               | Action dict missing required keys (validated against the manifest).                         |
| `IncompatibleVersionError` | r2d2 reports an API version that c3po does not support.                                     |

All exceptions subclass `RuntimeError` and include descriptive messages.

---

## 9. Internal Architecture

```
c3po/src/c3po/
├── __init__.py           # Exports: Robot, RecordingSession, exceptions
├── robot.py              # Public Robot class
├── _transport/
│   ├── __init__.py
│   └── _websocket.py     # WebSocket client with JSON text + binary frame support
├── _manifest.py          # Manifest parsing and observation key derivation
├── _buffer.py            # Thread‑safe observation buffer with blocking wait
├── _keyboard.py          # TerminalKeyListener (pure stdlib on Unix)
└── exceptions.py         # Custom exception classes
```

### Connection lifecycle

1. `Robot.__init__` opens a WebSocket to `ws://{host}:{port}`.
2. Sends a `describe` request and blocks for the manifest response (with timeout).
3. Parses the manifest. Populates properties (`arms`, `cameras`, `action_keys`, etc.).
4. Verifies API version compatibility.
5. A background thread reads incoming WebSocket messages:
   - JSON text frames (observations) → updates the observation buffer.
   - Binary frames (camera images) → JPEG‑decoded via Pillow → updates the buffer.
6. `reset()` sends a new‑episode signal to r2d2 and blocks until all observation keys have data.
7. `step()` sends the action (if any) as a JSON text frame, then blocks until the next observation arrives.
8. On context manager exit: sends `stop_recording` if recording, closes the WebSocket, stops the keyboard listener.

### Observation buffer

A thread‑safe latest‑value buffer. The WebSocket thread writes; the user's thread reads. `wait_next()` blocks until new data arrives after the last read — this is what synchronizes c3po's loop to r2d2's control cycle.

### Keyboard listener

A background thread reads raw bytes from the controlling TTY using `termios` + `tty` + `select`. It parses single‑byte keys (q, n, r) and sets flags on a shared `RecordingSession` object. On Windows, where `termios` is unavailable, the listener is a no‑op. No `pynput` dependency.

---

## 10. Versioning and Compatibility

c3po follows semantic versioning. Compatible with any r2d2 server reporting the same `api_version` major number. The `describe` manifest is the sole compatibility contract — as long as the manifest structure and protocol remain backward‑compatible within a major version, c3po and r2d2 can be upgraded independently.
