"""Thread-safe observation buffer.

Used internally by :class:`Robot` to collect data from incoming
WebSocket messages and deliver the latest available snapshot to
the user-facing ``step()`` call.

The buffer stores the latest value for each observation key (joint
state as ``np.ndarray``, camera frames decoded from JPEG or raw bytes).
``get()`` returns a snapshot dict without blocking.
"""

from __future__ import annotations

import logging
import threading
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from c3po._protocol import BinaryFrame

logger = logging.getLogger(__name__)


class ObservationBuffer:
    """Thread-safe latest-value buffer for observation data.

    Values are updated by :meth:`update` (joint state) and
    :meth:`update_from_frame` (camera images).  :meth:`get` returns
    the latest snapshot without blocking.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray] = {}
        self._event = threading.Event()
        self._last_frame_ids: dict[str, int] = {}

    def update(self, key: str, value: np.ndarray) -> None:
        """Store the latest value for *key*."""
        with self._lock:
            self._latest[key] = value
        self._event.set()

    def update_from_frame(self, frame: BinaryFrame) -> None:
        """Decode a camera frame and store it as a ``np.ndarray``.

        Handles JPEG, PNG, raw RGB, and raw depth encodings.

        Parameters
        ----------
        frame:
            A ``BinaryFrame`` received from the WebSocket transport.
        """
        from c3po._protocol import Encoding

        encoding = frame.encoding
        payload = frame.jpeg_bytes

        if encoding == Encoding.JPEG_RGB:
            from PIL import Image

            img = Image.open(BytesIO(payload))
            arr = np.array(img.convert("RGB"))
        elif encoding == Encoding.RAW_RGB:
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(
                frame.height, frame.width, 3
            )
        elif encoding == Encoding.RAW_DEPTH:
            arr = np.frombuffer(payload, dtype=np.uint16).reshape(
                frame.height, frame.width
            )
        elif encoding == Encoding.PNG_DEPTH:
            from PIL import Image

            img = Image.open(BytesIO(payload))
            arr = np.array(img).astype(np.uint16)
        else:
            logger.warning(
                "Unknown frame encoding %d for camera %r — skipping",
                encoding,
                frame.camera_name,
            )
            return

        # Track frame IDs for diagnostics.
        fid = frame.frame_id
        if fid > 0:
            last = self._last_frame_ids.get(frame.camera_name, 0)
            if last > 0 and fid > last + 1:
                logger.warning(
                    "Camera %r dropped %d frame(s) (last id=%d, current id=%d)",
                    frame.camera_name,
                    fid - last - 1,
                    last,
                    fid,
                )
            self._last_frame_ids[frame.camera_name] = fid

        with self._lock:
            self._latest[frame.camera_name] = arr
        self._event.set()

    def get(self) -> dict[str, np.ndarray]:
        """Return a snapshot of the latest values for all keys.

        Does not block.  Returns an empty dict if no data has been
        received yet.
        """
        with self._lock:
            return dict(self._latest)

    def wait_until_any(self, timeout: float | None = None) -> bool:
        """Block until at least one observation key has been updated.

        Returns ``True`` if data arrived, ``False`` on timeout.
        """
        result = self._event.wait(timeout=timeout)
        if result:
            self._event.clear()
        return result

    def clear(self) -> None:
        """Clear all stored values."""
        with self._lock:
            self._latest.clear()
        self._event.clear()
