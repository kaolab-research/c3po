"""WebSocket transport using the N-Droids custom protocol.

Communicates with r2d2 over a single WebSocket connection.
Sends JSON text frames for control and receives both JSON text
and binary JPEG frames from the server.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import websocket

from c3po._protocol import (
    BinaryFrame,
    decode_binary_frame,
    decode_json_message,
    encode_json_message,
)
from c3po._transport._base import Transport
from c3po.exceptions import RobotDisconnectedError, RobotNotFoundError

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0
_RECV_POLL_INTERVAL = 0.1


class WebSocketTransport(Transport):
    """Transport that speaks the N-Droids WebSocket protocol.

    Parameters
    ----------
    url:
        WebSocket server URL, e.g. ``"ws://10.42.0.1:9090"``.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._recv_queue: queue.Queue[Any] = queue.Queue()
        self._connected = threading.Event()
        self._error: Exception | None = None

    # ------------------------------------------------------------------
    # Transport interface
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def connect(self) -> None:
        """Open the WebSocket and block until the connection is established.

        Raises
        ------
        RobotNotFoundError
            If the connection cannot be established within the timeout.
        """
        self._ws = websocket.WebSocketApp(
            self._url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self._thread = threading.Thread(
            target=self._ws.run_forever,
            daemon=True,
            name="c3po-ws",
        )
        self._thread.start()

        if not self._connected.wait(timeout=_CONNECT_TIMEOUT):
            self.close()
            err = self._error
            detail = f": {err}" if err else ""
            raise RobotNotFoundError(
                f"Could not connect to r2d2 at {self._url} within {_CONNECT_TIMEOUT} s."
                f"{detail}"
            )

    def close(self) -> None:
        """Close the WebSocket and join the receive thread."""
        self._connected.clear()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
            self._ws = None
        # Put a sentinel to unblock any waiting recv() call.
        try:
            self._recv_queue.put_nowait(_ConnectionClosed())
        except queue.Full:
            pass

    def send_json(self, msg: Any) -> None:
        """Encode *msg* as a JSON text frame and send it."""
        if self._ws is None or not self.is_connected:
            raise RobotDisconnectedError("Not connected.")
        raw = encode_json_message(msg)
        self._ws.send(raw)

    def send_binary(self, data: bytes) -> None:
        """Send raw *data* as a WebSocket binary frame."""
        if self._ws is None or not self.is_connected:
            raise RobotDisconnectedError("Not connected.")
        self._ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)

    def recv(self, timeout: float | None = None) -> Any:
        """Return the next message from the server.

        Blocks until a message is available or *timeout* expires.

        Raises
        ------
        RobotDisconnectedError
            If the connection is lost or the timeout expires.
        """
        try:
            msg = self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise RobotDisconnectedError(f"No message received within {timeout} s.")

        if isinstance(msg, _ConnectionClosed):
            raise RobotDisconnectedError("Connection to r2d2 was closed.")

        return msg

    # ------------------------------------------------------------------
    # WebSocket callbacks (run in ws thread)
    # ------------------------------------------------------------------

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        logger.debug("WebSocket connected to %s", self._url)
        self._connected.set()
        self._error = None

    def _on_message(self, ws: websocket.WebSocketApp, raw: str | bytes) -> None:
        """Called for both text and binary frames.

        *raw* is ``str`` for text frames and ``bytes`` for binary frames
        (``websocket-client`` >= 1.4 dispatches both through ``on_message``).
        """
        try:
            if isinstance(raw, str):
                msg = decode_json_message(raw)
            else:
                msg = decode_binary_frame(raw)
            self._recv_queue.put(msg)
        except Exception as exc:
            logger.warning("Failed to decode incoming message: %s", exc)

    def _on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        logger.error("WebSocket error: %s", error)
        self._error = error if isinstance(error, Exception) else Exception(str(error))

    def _on_close(
        self, ws: websocket.WebSocketApp, close_status_code: Any, close_msg: Any
    ) -> None:
        logger.debug("WebSocket closed: %s %s", close_status_code, close_msg)
        self._connected.clear()
        # Put a sentinel to unblock recv().
        try:
            self._recv_queue.put_nowait(_ConnectionClosed())
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Internal sentinel
# ---------------------------------------------------------------------------


class _ConnectionClosed:
    """Sentinel placed on the recv queue when the connection closes."""
