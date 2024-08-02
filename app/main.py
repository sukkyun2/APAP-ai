import asyncio
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.api_response import ApiResponse
from app.config import settings
from app.connection_manager import ConnectionManager
from app.history import async_save_history
from model.detect import detect, track

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins.split(';'),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()


# video_recorder = VideoRecorder()


@app.post("/api/detect-image", response_model=ApiResponse)
async def detect_image(file: UploadFile = File(...)) -> ApiResponse:
    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as err:
        return ApiResponse.bad_request(str(err))

    result = detect(np.array(img))
    await async_save_history(result)

    return ApiResponse.ok()


@app.get("/api/exists-publisher")
def exists_publisher() -> ApiResponse[bool]:
    return ApiResponse[bool].ok_with_data(bool(manager.publisher))


@app.websocket("/ws/publisher")
async def websocket_publisher(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # byte to nparr

            result = track(img)
            await async_save_history(result)

            # TODO 별도 이상상황으로 교체
            cell_phone_detected = any(det.class_name == 'cell phone' for det in result.detections)
            if cell_phone_detected:
                await async_save_history(result)
            # video_recorder.save_frame(result.predict_image_np)

            await manager.broadcast(result.get_encoded_nparr().tobytes())
    except WebSocketDisconnect:
        manager.disconnect()
        print("Publisher disconnected")


@app.websocket("/ws/subscriber")
async def websocket_subscriber(websocket: WebSocket):
    await manager.subscribe(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("Subscriber disconnected")
    finally:
        manager.unsubscribe(websocket)
