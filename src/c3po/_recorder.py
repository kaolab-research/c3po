"""Recording and replay for c3po sessions.

Provides :class:`Recorder` for saving observations, actions, rewards,
and terminal flags to HDF5, and :class:`ReplayClient` for offline
replay of recorded sessions.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class Recorder:
    """Records observation/action/reward/done tuples to an HDF5 file.

    Usage::

        with Robot() as robot, Recorder(robot, "session.h5") as rec:
            obs = robot.reset()
            rec.start_episode()
            while not done:
                action = policy(obs)
                obs = robot.step(action)
                rec.step(obs, action, reward, done)
            rec.end_episode()
    """

    def __init__(self, robot: Any, path: str) -> None:
        self._robot = robot
        self._path = path
        self._file: Any = None
        self._episode_idx = 0
        self._step_idx = 0
        # Defer h5py import so it's only required when recording.
        import h5py

        self._h5py = h5py
        self._obs_keys = (
            self._robot._obs_keys if hasattr(self._robot, "_obs_keys") else []
        )

    def __enter__(self) -> Recorder:
        self._file = self._h5py.File(self._path, "w")
        # Write manifest as file attribute.
        manifest = {
            "station_model": getattr(self._robot, "station_model", "unknown"),
            "control_rate": getattr(self._robot, "control_rate", 0.0),
            "arms": getattr(self._robot, "arms", []),
            "cameras": getattr(self._robot, "cameras", []),
            "controllers": getattr(self._robot, "controllers", []),
            "action_keys": getattr(self._robot, "action_keys", []),
        }
        self._file.attrs["manifest"] = json.dumps(manifest)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def start_episode(self) -> None:
        """Mark the start of a new episode."""
        self._step_idx = 0
        grp = self._file.require_group(f"episode_{self._episode_idx}")
        grp.attrs["step_count"] = 0

    def step(
        self,
        obs: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        reward: float = 0.0,
        done: bool = False,
    ) -> None:
        """Record one transition."""
        grp = self._file[f"episode_{self._episode_idx}"]
        step_grp = grp.require_group(f"step_{self._step_idx}")
        obs_grp = step_grp.require_group("observation")
        for key, val in obs.items():
            obs_grp.create_dataset(key, data=val, compression="gzip")
        act_grp = step_grp.require_group("action")
        for key, val in action.items():
            act_grp.create_dataset(key, data=val, compression="gzip")
        step_grp.attrs["reward"] = reward
        step_grp.attrs["done"] = done
        self._step_idx += 1
        grp.attrs["step_count"] = self._step_idx

    def end_episode(self) -> None:
        """Mark the end of the current episode."""
        self._episode_idx += 1


class ReplayClient:
    """Replays a recorded HDF5 session.

    Exposes the same ``Robot`` API for offline policy evaluation.
    """

    def __init__(self, path: str) -> None:
        import h5py

        self._file: Any = h5py.File(path, "r")
        manifest_raw = self._file.attrs.get("manifest", "{}")
        self._manifest = (
            json.loads(manifest_raw) if isinstance(manifest_raw, str) else {}
        )
        self._episode_idx = 0
        self._step_idx = 0
        self._closed = False

    # -- Robot-compatible properties -----------------------------------------

    @property
    def station_model(self) -> str:
        return self._manifest.get("station_model", "replay")

    @property
    def control_rate(self) -> float:
        return self._manifest.get("control_rate", 50.0)

    @property
    def arms(self) -> list[str]:
        return self._manifest.get("arms", [])

    @property
    def cameras(self) -> list[str]:
        return self._manifest.get("cameras", [])

    @property
    def controllers(self) -> list[str]:
        return self._manifest.get("controllers", [])

    @property
    def action_keys(self) -> list[str]:
        return self._manifest.get("action_keys", [])

    # -- Robot-compatible methods --------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        self._step_idx = 0
        return self._read_obs()

    def step(
        self, action: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        self._step_idx += 1
        return self._read_obs()

    def close(self) -> None:
        if not self._closed and self._file is not None:
            self._file.close()
            self._closed = True

    def __enter__(self) -> ReplayClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- iteration -----------------------------------------------------------

    def episodes(self):
        """Yield (obs, action, reward, done) tuples for all episodes."""
        ep = 0
        while f"episode_{ep}" in self._file:
            grp = self._file[f"episode_{ep}"]
            count = grp.attrs.get("step_count", 0)
            for i in range(count):
                step_grp = grp[f"step_{i}"]
                obs = {}
                if "observation" in step_grp:
                    for key in step_grp["observation"]:
                        obs[key] = np.array(step_grp[f"observation/{key}"])
                action = {}
                if "action" in step_grp:
                    for key in step_grp["action"]:
                        action[key] = np.array(step_grp[f"action/{key}"])
                reward = step_grp.attrs.get("reward", 0.0)
                done = step_grp.attrs.get("done", False)
                yield obs, action, reward, done
            ep += 1

    def __iter__(self):
        return self.episodes()

    def __del__(self) -> None:
        self.close()

    # -- internal ------------------------------------------------------------

    def _read_obs(self) -> dict[str, np.ndarray]:
        grp = self._file[f"episode_{self._episode_idx}"]
        step_grp = grp[f"step_{self._step_idx}"]
        obs: dict[str, np.ndarray] = {}
        if "observation" in step_grp:
            for key in step_grp["observation"]:
                obs[key] = np.array(step_grp[f"observation/{key}"])
        return obs
