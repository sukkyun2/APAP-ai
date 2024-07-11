from typing import List

from starlette.websockets import WebSocket


class PublisherAlreadyExistsError(Exception):
    def __init__(self, message="이미 publisher가 존재합니다"):
        self.message = message
        super().__init__(self.message)


class ConnectionManager:
    def __init__(self):
        self.publisher = None
        self.subscribers: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        if self.publisher:
            raise PublisherAlreadyExistsError()

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
            await subscriber.send_bytes(message)
