"""N-Droids wire protocol — JSON message types and binary frame encoding.

This module defines every message that crosses the WebSocket between c3po
and r2d2.  It has zero dependencies outside the standard library and is
duplicated in both repositories (they change rarely and independently).

.. warning::

    This file is manually duplicated in c3po and r2d2.  Any change here
    MUST be mirrored in the other repository's copy.  The two copies are
    the protocol contract — divergence causes silent incompatibility.
    Consider running ``diff r2d2/src/r2d2/_protocol.py c3po/src/c3po/_protocol.py``
    after making changes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProtocolError(ValueError):
    """Raised when a protocol message cannot be decoded."""


# ---------------------------------------------------------------------------
# Binary frame encoding constants
# ---------------------------------------------------------------------------


class Encoding:
    """Encoding constants for camera frame binary payloads."""

    JPEG_RGB: ClassVar[int] = 0
    JPEG_DEPTH: ClassVar[int] = 1
    PNG_DEPTH: ClassVar[int] = 2
    RAW_RGB: ClassVar[int] = 3
    RAW_DEPTH: ClassVar[int] = 4


_HEADER_STRUCT = struct.Struct(">I")  # name length
_META_STRUCT = struct.Struct(">QIIBI")  # ts_ns, width, height, encoding, frame_id
_HEADER_OVERHEAD = _HEADER_STRUCT.size + _META_STRUCT.size  # 4 + 8+4+4+1+4 = 25


# ---------------------------------------------------------------------------
# JSON message types
#
# Each message is a dataclass with a ``type`` field that is constant for
# that message kind (not passed to the constructor — ``init=False``).
# ``encode_json_message`` serialises to a JSON string.
# ``decode_json_message`` deserialises from a JSON string and dispatches
# to the correct dataclass based on the ``type`` field.
# ---------------------------------------------------------------------------


@dataclass
class DescribeRequest:
    """Sent by c3po to request the station manifest."""

    type: str = field(default="describe", init=False)


@dataclass
class DescribeResponse:
    """Sent by r2d2 in response to a describe request."""

    manifest: dict[str, Any]
    type: str = field(default="describe_response", init=False)


@dataclass
class Action:
    """Sent by c3po to command the robot.

    ``data`` maps action keys (e.g. ``"follower/joint_position"``) to
    lists of float64 values.  ``reward`` and ``done`` are optional
    dataset metadata — they do not affect the control loop.
    """

    data: dict[str, list[float]]
    reward: float = 0.0
    done: bool = False
    type: str = field(default="action", init=False)


@dataclass
class Observation:
    """Sent by r2d2 to stream the latest joint state and other scalar
    sensor readings.

    Camera frames are sent as separate binary frames, not in this message.
    """

    data: dict[str, list[float]]
    type: str = field(default="observation", init=False)


@dataclass
class ResetEpisode:
    """Sent by c3po to mark the start of a new episode (for recording)."""

    type: str = field(default="reset_episode", init=False)


@dataclass
class Configure:
    """Sent by c3po to request a control rate."""

    rate: float
    type: str = field(default="configure", init=False)


@dataclass
class StartRecording:
    """Sent by c3po to tell r2d2 to begin recording a dataset."""

    name: str
    type: str = field(default="start_recording", init=False)


@dataclass
class StopRecording:
    """Sent by c3po to tell r2d2 to finalize and close the dataset."""

    type: str = field(default="stop_recording", init=False)


@dataclass
class CancelEpisode:
    """Sent by c3po to tell r2d2 to discard the current in-progress episode."""

    type: str = field(default="cancel_episode", init=False)


# Registry: ``type`` field value → dataclass constructor.
# Used by ``decode_json_message`` for dispatch.
_MESSAGE_REGISTRY: dict[str, type] = {
    "describe": DescribeRequest,
    "describe_response": DescribeResponse,
    "action": Action,
    "observation": Observation,
    "reset_episode": ResetEpisode,
    "configure": Configure,
    "start_recording": StartRecording,
    "stop_recording": StopRecording,
    "cancel_episode": CancelEpisode,
}


# ---------------------------------------------------------------------------
# JSON encode / decode
# ---------------------------------------------------------------------------


def encode_json_message(msg: Any) -> str:
    """Encode a protocol message dataclass to a JSON string.

    Parameters
    ----------
    msg:
        Any dataclass defined in this module (e.g. ``DescribeRequest()``).

    Returns
    -------
    str
        JSON string with ``type`` and all other fields.
    """
    from dataclasses import asdict

    return json.dumps(asdict(msg))


def decode_json_message(raw: str) -> Any:
    """Decode a JSON string to the appropriate protocol message dataclass.

    Dispatches on the ``type`` field using the internal message registry.

    Parameters
    ----------
    raw:
        JSON string received from the WebSocket.

    Returns
    -------
    A dataclass instance (e.g. ``Observation``, ``Action``).

    Raises
    ------
    ProtocolError
        If ``type`` is missing or unknown, or if required fields are absent.
    """
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON: {exc}") from exc

    msg_type = data.get("type")
    if not msg_type:
        raise ProtocolError("Message missing required 'type' field")

    cls = _MESSAGE_REGISTRY.get(msg_type)
    if cls is None:
        raise ProtocolError(f"Unknown message type: {msg_type!r}")

    # Remove 'type' — it is auto-set by the dataclass default.
    kwargs = {k: v for k, v in data.items() if k != "type"}

    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ProtocolError(
            f"Invalid fields for message type {msg_type!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Binary frame encode / decode
# ---------------------------------------------------------------------------


@dataclass
class BinaryFrame:
    """A single camera frame sent as a binary WebSocket message.

    The wire format is::

        [4 B  name_len  ]  uint32 big-endian
        [N B  name      ]  UTF-8 encoded camera name
        [8 B  ts_ns     ]  uint64 big-endian — capture timestamp in nanoseconds
        [4 B  width     ]  uint32 big-endian
        [4 B  height    ]  uint32 big-endian
        [1 B  encoding  ]  0=JPEG_RGB, 1=JPEG_DEPTH, 2=PNG_DEPTH, 3=RAW_RGB, 4=RAW_DEPTH
        [4 B  frame_id  ]  uint32 big-endian — monotonic per camera, 0 if unused
        [M B  payload]    raw payload (JPEG bytes or raw numpy bytes)
    """

    camera_name: str
    timestamp_ns: int
    width: int
    height: int
    encoding: int
    payload: bytes
    frame_id: int = 0


def encode_binary_frame(frame: BinaryFrame) -> bytes:
    """Encode a ``BinaryFrame`` to wire-format bytes."""
    name_bytes = frame.camera_name.encode("utf-8")
    header = _HEADER_STRUCT.pack(len(name_bytes))
    header += name_bytes
    header += _META_STRUCT.pack(
        frame.timestamp_ns, frame.width, frame.height, frame.encoding, frame.frame_id
    )
    return header + frame.payload


def decode_binary_frame(data: bytes) -> BinaryFrame:
    """Decode wire-format bytes to a ``BinaryFrame``.

    Raises
    ------
    ProtocolError
        If *data* is too short to contain a complete header.
    """
    if len(data) < _HEADER_OVERHEAD:
        raise ProtocolError(
            f"Binary frame too short: {len(data)} bytes (minimum is {_HEADER_OVERHEAD})"
        )

    name_len = _HEADER_STRUCT.unpack(data[0:4])[0]
    offset = 4

    if len(data) < offset + name_len + _META_STRUCT.size:
        raise ProtocolError(
            f"Binary frame header truncated: need at least "
            f"{offset + name_len + _META_STRUCT.size} bytes, got {len(data)}"
        )

    camera_name = data[offset : offset + name_len].decode("utf-8")
    offset += name_len

    ts_ns, width, height, encoding, frame_id = _META_STRUCT.unpack(
        data[offset : offset + _META_STRUCT.size]
    )
    offset += _META_STRUCT.size

    payload = data[offset:]

    return BinaryFrame(
        camera_name=camera_name,
        timestamp_ns=ts_ns,
        width=width,
        height=height,
        encoding=encoding,
        payload=payload,
        frame_id=frame_id,
    )
