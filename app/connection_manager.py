from typing import List

from starlette.websockets import WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        self.publisher = None
        self.subscribers: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.publisher = websocket

    def disconnect(self):
        self.publisher = None

    async def subscribe(self, websocket: WebSocket):
        await websocket.accept()
        self.subscribers.append(websocket)

    def unsubscribe(self, websocket: WebSocket):
        self.subscribers.remove(websocket)

    async def broadcast(self, message: bytes):
        for subscriber in self.subscribers:
            try:
                await subscriber.send_bytes(message)
            except WebSocketDisconnect:
                self.unsubscribe(subscriber)
