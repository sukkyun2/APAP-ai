import asyncio
import io
import os
import tempfile
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Query, Form

from app.api_response import ApiResponse, ApiListResponse
from app.config import settings
from app.connection_manager import ConnectionManager
from app.history import async_save_history
from model.detect import detect
from model.operations import OperationType, define_operation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins.split(';'),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()


@app.post("/api/detect-image", response_model=ApiResponse)
async def detect_image(file: UploadFile = File(...)) -> ApiResponse:
    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as err:
        return ApiResponse.bad_request(str(err))

    result = detect(np.array(img))
    await async_save_history(result, "NONE")

    return ApiResponse.ok()


@app.get("/api/publishers")
def exists_publisher() -> ApiListResponse[str]:
    return ApiListResponse[str].ok_with_data(list(manager.publishers.keys()))


@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...),
                       location_name: str = Form(...),
                       op: Optional[OperationType] = Form(...)):
    file_content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)

    if not cap.isOpened():
        os.remove(temp_file_path)
        return ApiResponse.bad_request("Error opening video file")

    async def process_video():
        while True:
            ret, img = cap.read()
            if not ret:
                break

            operation = define_operation(op)
            pattern_detected, result = operation(img)
            if pattern_detected:
                await async_save_history(result, location_name)

            await manager.broadcast(location_name, result.get_encoded_nparr().tobytes())

        cap.release()
        os.remove(temp_file_path)

    asyncio.create_task(process_video())

    return ApiResponse.ok()


@app.websocket("/ws/publishers/{location_name}")
async def websocket_publisher(websocket: WebSocket,
                              location_name: str,
                              op: Optional[OperationType] = Query(OperationType.ESTIMATE_DISTANCE)):
    await manager.connect(location_name, websocket)

    try:
        while True:
            # pre-processing
            data = await websocket.receive_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # byte to nparr

            operation = define_operation(op)
            pattern_detected, result = operation(img)
            if pattern_detected:
                await async_save_history(result, location_name)

            await manager.broadcast(location_name, result.get_encoded_nparr().tobytes())
    except WebSocketDisconnect:
        manager.disconnect(location_name)
        print("Publisher disconnected")


@app.websocket("/ws/subscribers/{location_name}")
async def websocket_subscriber(location_name: str, websocket: WebSocket):
    await manager.subscribe(location_name, websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("Subscriber disconnected")
    finally:
        manager.unsubscribe(location_name, websocket)
