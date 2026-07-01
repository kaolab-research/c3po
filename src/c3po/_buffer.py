"""Thread-safe observation buffer.

Used internally by :class:`Robot` to collect data from incoming
WebSocket messages and deliver the latest available snapshot to
the user-facing ``step()`` call.

The buffer stores the latest value for each observation key (joint
state as ``np.ndarray``, camera frames decoded from JPEG bytes).
``get()`` returns a snapshot dict without blocking.
"""

from __future__ import annotations

import threading
from io import BytesIO

import numpy as np
from PIL import Image


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

    def update(self, key: str, value: np.ndarray) -> None:
        """Store the latest value for *key*."""
        with self._lock:
            self._latest[key] = value
        self._event.set()

    def update_from_frame(self, camera_name: str, jpeg_bytes: bytes) -> None:
        """Decode a JPEG camera frame and store it as an RGB ``np.ndarray``.

        Parameters
        ----------
        camera_name:
            Observation key for this camera (e.g. ``"front_rgb"``).
        jpeg_bytes:
            Raw JPEG image bytes.
        """
        img = Image.open(BytesIO(jpeg_bytes))
        arr = np.array(img.convert("RGB"))
        with self._lock:
            self._latest[camera_name] = arr
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
