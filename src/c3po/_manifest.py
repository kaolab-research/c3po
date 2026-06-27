"""Manifest parsing — translates the /describe service response into
structured metadata that the Robot class exposes to the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse a raw /describe response dict into a structured manifest.

    Parameters
    ----------
    raw:
        The ``values`` dict from a rosbridge service call response.

    Returns
    -------
    dict
        Structured manifest with keys: ``api_version``, ``station_model``,
        ``control_rate``, ``arms``, ``cameras``, ``controllers``,
        ``action_keys``, ``joint_names``.
    """
    manifest_json = raw.get("manifest_json", "{}")
    try:
        manifest = json.loads(manifest_json)
    except json.JSONDecodeError:
        logger.warning("Failed to parse manifest_json — using defaults")
        manifest = {}

    # -- top-level fields ----------------------------------------------------
    # manifest_json is authoritative; raw fields are fallbacks.
    api_version = str(manifest.get("api_version", "0.0"))
    station_model = str(
        manifest.get("station_model") or raw.get("robot_model", "unknown")
    )
    control_rate = float(manifest.get("control_rate") or raw.get("control_rate", 50.0))

    # -- arms ----------------------------------------------------------------
    arms: list[dict[str, Any]] = manifest.get("arms", [])
    if not arms:
        # Backward compat: single arm from legacy flat fields.
        joint_count = int(raw.get("joint_count", 0))
        if joint_count > 0:
            arms = [
                {
                    "name": "",
                    "joint_count": joint_count,
                    "joint_names": list(raw.get("joint_names", [])),
                    "command_mode": str(raw.get("command_mode", "joint_position")),
                }
            ]

    arm_names: list[str] = [a["name"] for a in arms]

    # -- cameras -------------------------------------------------------------
    cameras: list[dict[str, Any]] = manifest.get("cameras", [])
    if not cameras:
        # Backward compat: flat camera_names list.
        for name in raw.get("camera_names", []):
            cameras.append({"name": name})

    camera_names: list[str] = [c["name"] for c in cameras]

    # -- controllers ---------------------------------------------------------
    controllers: list[dict[str, Any]] = manifest.get("controllers", [])
    controller_names: list[str] = [c["name"] for c in controllers]

    # -- action keys ---------------------------------------------------------
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
