import asyncio
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import model.llm_api
from app.api_response import ApiResponse
from app.config import settings
from app.connection_manager import ConnectionManager
from app.history import HistorySaveRequest, save_history
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


@app.post("/api/detect-image", response_model=ApiResponse)
@app.post("/detect-image", response_model=ApiResponse)  # TODO 추후 API 교체 후 제거
async def detect_image(file: UploadFile = File(...)) -> ApiResponse:
    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as err:
        return ApiResponse.bad_request(str(err))

    result = detect(np.array(img))
    asyncio.create_task(
        save_history(HistorySaveRequest(image=result.get_image(), detections=result.detections))
    )

    print(model.llm_api.call_gemini(img, result.detections))

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
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            result = track(img)

            _, buffer = cv2.imencode('.jpg', result.predict_image_np)
            processed_bytes = buffer.tobytes()

            await manager.broadcast(processed_bytes)
    except WebSocketDisconnect:
        if not manager.subscribers:
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
