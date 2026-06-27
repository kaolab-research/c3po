"""WebSocket transport using the rosbridge protocol.

Communicates with r2d2's rosbridge_server over a single WebSocket connection.
Implements the standard rosbridge JSON protocol for topic subscribe/publish
and service calls.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any, Callable

import websocket
from c3po._transport._base import Transport

from c3po.exceptions import RobotDisconnectedError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rosbridge protocol constants
# ---------------------------------------------------------------------------

_ROSBRIDGE_OP_SUBSCRIBE = "subscribe"
_ROSBRIDGE_OP_UNSUBSCRIBE = "unsubscribe"
_ROSBRIDGE_OP_PUBLISH = "publish"
_ROSBRIDGE_OP_CALL_SERVICE = "call_service"
_ROSBRIDGE_OP_SERVICE_RESPONSE = "service_response"


# ---------------------------------------------------------------------------
# WebSocket transport
# ---------------------------------------------------------------------------


class WebSocketTransport(Transport):
    """Transport that speaks the rosbridge WebSocket protocol."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._subscriptions: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._pending_services: dict[str, threading.Event] = {}
        self._service_results: dict[str, dict[str, Any]] = {}
        self._connected = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Transport interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._ws = websocket.WebSocketApp(
            self._url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever, daemon=True, name="c3po-ws"
        )
        self._thread.start()

        # Block until connected or timeout.
        if not self._connected.wait(timeout=10.0):
            raise RobotDisconnectedError(
                f"Failed to connect to {self._url} within 10 s. "
                "Is the Ethernet cable plugged in?"
            )

    def close(self) -> None:
        ws = self._ws
        if ws is not None:
            ws.close()
            self._ws = None
        self._thread = None
        self._connected.clear()

    def call_service(
        self, service: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._ws is None:
            raise RobotDisconnectedError("Not connected.")

        service_id = f"call_service_{uuid.uuid4().hex}"
        event = threading.Event()

        with self._lock:
            self._pending_services[service_id] = event

        msg = {
            "op": _ROSBRIDGE_OP_CALL_SERVICE,
            "id": service_id,
            "service": service,
            "args": request or {},
        }
        self._ws.send(json.dumps(msg))

        if not event.wait(timeout=10.0):
            raise RobotDisconnectedError(
                f"Service call {service} timed out after 10 s."
            )

        with self._lock:
            result = self._service_results.pop(service_id, {})

        if "values" not in result:
            raise RobotDisconnectedError(f"Service call {service} returned no values.")

        return result["values"]

    def subscribe(
        self, topic: str, msg_type: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        with self._lock:
            self._subscriptions[topic] = callback

        if self._ws is None:
            return

        msg = {
            "op": _ROSBRIDGE_OP_SUBSCRIBE,
            "id": f"subscribe_{topic}",
            "topic": topic,
            "type": msg_type,
        }
        self._ws.send(json.dumps(msg))
        logger.debug("Subscribed to %s (%s)", topic, msg_type)

    def unsubscribe(self, topic: str) -> None:
        with self._lock:
            self._subscriptions.pop(topic, None)

        if self._ws is None:
            return

        msg = {
            "op": _ROSBRIDGE_OP_UNSUBSCRIBE,
            "id": f"unsubscribe_{topic}",
            "topic": topic,
        }
        self._ws.send(json.dumps(msg))

    def publish(self, topic: str, msg_type: str, msg: dict[str, Any]) -> None:
        if self._ws is None:
            raise RobotDisconnectedError("Not connected.")

        payload = {
            "op": _ROSBRIDGE_OP_PUBLISH,
            "topic": topic,
            "msg": msg,
        }
        self._ws.send(json.dumps(payload))

    # ------------------------------------------------------------------
    # WebSocket callbacks (run in ws thread)
    # ------------------------------------------------------------------

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        logger.debug("WebSocket connected to %s", self._url)
        self._connected.set()

        # Re-subscribe to any topics registered before connect completed.
        with self._lock:
            for topic in list(self._subscriptions):
                msg = {
                    "op": _ROSBRIDGE_OP_SUBSCRIBE,
                    "id": f"subscribe_{topic}",
                    "topic": topic,
                    "type": "",  # type was stored... skip for simplicity
                }
                ws.send(json.dumps(msg))

    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON from rosbridge")
            return

        op = data.get("op")

        if op == _ROSBRIDGE_OP_PUBLISH:
            topic = data.get("topic", "")
            msg = data.get("msg", {})
            with self._lock:
                cb = self._subscriptions.get(topic)
            if cb is not None:
                cb(msg)

        elif op == _ROSBRIDGE_OP_SERVICE_RESPONSE:
            sid = data.get("id", "")
            with self._lock:
                self._service_results[sid] = data
                event = self._pending_services.pop(sid, None)
            if event is not None:
                event.set()

    def _on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        logger.error("WebSocket error: %s", error)

    def _on_close(
        self, ws: websocket.WebSocketApp, close_status_code: Any, close_msg: Any
    ) -> None:
        logger.warning("WebSocket closed: %s %s", close_status_code, close_msg)
        self._connected.clear()
