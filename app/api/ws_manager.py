from __future__ import annotations

from datetime import datetime, timezone

from fastapi import WebSocket
from loguru import logger


class WSManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, event: str, data: dict):
        payload = {"event": event, "data": data, "timestamp": datetime.now(timezone.utc).isoformat()}
        for ws in list(self.connections):
            try:
                await ws.send_json(payload)
            except Exception as exc:
                logger.warning(f"Removing dead ws connection: {exc}")
                self.disconnect(ws)
