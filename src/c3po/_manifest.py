"""Manifest parsing — translates the /describe manifest into structured
metadata that the :class:`Robot` class exposes to the user.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Parse a manifest dict (from the ``describe_response`` message) into
    a structured metadata dict.

    Parameters
    ----------
    manifest:
        The ``manifest`` field from a ``DescribeResponse`` message.

    Returns
    -------
    dict
        Structured metadata with keys: ``api_version``, ``station_model``,
        ``control_rate``, ``arm_names``, ``camera_names``, ``controller_names``,
        ``action_keys``, ``joint_names``, plus the original ``arms``,
        ``cameras``, and ``controllers`` lists.
    """
    api_version = str(manifest.get("api_version", "0.0"))
    station_model = str(manifest.get("station_model", "unknown"))
    control_rate = float(manifest.get("control_rate", 50.0))

    arms: list[dict[str, Any]] = manifest.get("arms", [])
    arm_names: list[str] = [a["name"] for a in arms]

    cameras: list[dict[str, Any]] = manifest.get("cameras", [])
    camera_names: list[str] = [c["name"] for c in cameras]

    controllers: list[dict[str, Any]] = manifest.get("controllers", [])
    controller_names: list[str] = [c["name"] for c in controllers]

    action_keys: list[str] = []
    joint_names: dict[str, list[str]] = {}
    for arm in arms:
        name = arm["name"]
        mode = arm.get("command_mode", "joint_position")
        prefix = f"{name}/" if name else ""
        key = f"{prefix}{mode}"
        action_keys.append(key)
        joint_names[prefix.rstrip("/") or "default"] = list(arm.get("joint_names", []))

    return {
        "api_version": api_version,
        "station_model": station_model,
        "control_rate": control_rate,
        "arm_names": arm_names,
        "camera_names": camera_names,
        "controller_names": controller_names,
        "action_keys": action_keys,
        "joint_names": joint_names,
        "arms": arms,
        "cameras": cameras,
        "controllers": controllers,
    }


def build_observation_keys(manifest: dict[str, Any]) -> list[str]:
    """Return the list of observation dictionary keys for this station."""
    keys: list[str] = []

    for arm in manifest["arms"]:
        name = arm["name"]
        prefix = f"{name}/" if name else ""
        keys.append(f"{prefix}joint_position")
        keys.append(f"{prefix}joint_velocity")

    for cam in manifest["cameras"]:
        keys.append(cam["name"])

    for ctrl in manifest["controllers"]:
        ctrl_type = ctrl.get("type", "")
        if ctrl_type == "joint_position":
            keys.append(f"{ctrl['name']}/joint_position")
        else:
            keys.append(f"{ctrl['name']}/state")

    return keys
