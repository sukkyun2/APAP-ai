from collections import defaultdict

from starlette.websockets import WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        self.publishers: defaultdict = defaultdict(WebSocket)
        self.subscribers: defaultdict = defaultdict(list)

    async def connect(self, location_name: str, websocket: WebSocket):
        await websocket.accept()
        self.publishers[location_name] = websocket

    def disconnect(self, location_name: str):
        del self.publishers[location_name]

    async def subscribe(self, location_name: str, websocket: WebSocket):
        await websocket.accept()
        self.subscribers[location_name].append(websocket)

    def unsubscribe(self, location_name: str, websocket: WebSocket):
        self.subscribers[location_name].remove(websocket)

    async def broadcast(self, location_name: str, message: bytes):
        subscribers_by_location = self.subscribers[location_name]
        for subscriber in subscribers_by_location:
            try:
                await subscriber.send_bytes(message)
            except WebSocketDisconnect:
                self.unsubscribe(location_name, subscriber)
