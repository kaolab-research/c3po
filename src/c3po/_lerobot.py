"""LeRobot format conversion utilities.

Converts c3po observation dictionaries to the LeRobot dataset format
for compatibility with LeRobot's training pipelines.
"""

from __future__ import annotations

import numpy as np


def to_lerobot_format(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert a c3po observation dict to LeRobot dataset format.

    Maps:
      ``joint_position`` → ``observation.state``
      ``{arm}/joint_position`` → ``observation.state`` (concatenated)
      ``{camera}_rgb`` → ``observation.images.{camera}``
      ``{camera}_depth`` → ``observation.images.{camera}_depth``
      ``{controller}/joint_position`` → (skipped — not part of state)
    """
    result: dict[str, np.ndarray] = {}

    # Collect joint positions from all arms into a single state vector.
    state_parts: list[np.ndarray] = []
    for key in sorted(obs):
        if key.endswith("/joint_position") or key == "joint_position":
            state_parts.append(obs[key].ravel())
    if state_parts:
        result["observation.state"] = np.concatenate(state_parts)

    # Map camera images.
    for key, value in obs.items():
        if key.endswith("_rgb"):
            cam_name = key[:-4]  # strip _rgb
            # Replace / with . for LeRobot naming convention.
            cam_name = cam_name.replace("/", ".")
            result[f"observation.images.{cam_name}"] = value
        elif key.endswith("_depth"):
            cam_name = key[:-6]  # strip _depth
            cam_name = cam_name.replace("/", ".")
            result[f"observation.images.{cam_name}_depth"] = value

    return result


def to_lerobot_action(action: dict[str, np.ndarray]) -> np.ndarray:
    """Convert a c3po action dict to a LeRobot action vector.

    Concatenates all action values into a single 1D array.
    """
    parts: list[np.ndarray] = []
    for key in sorted(action):
        parts.append(action[key].ravel())
    return np.concatenate(parts) if parts else np.array([])
