import asyncio
from io import BytesIO

from PIL import Image
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.api_response import ApiResponse
from app.config import settings
from app.history import HistorySaveRequest, save_history
from model.detect import detect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins.split(';'),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect-image", response_model=ApiResponse)
async def detect_image(file: UploadFile = File(...)) -> ApiResponse:
    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as err:
        return ApiResponse.bad_request(str(err))

    result = detect(img)
    asyncio.create_task(
        save_history(HistorySaveRequest(image=result.predicted_image, detections=result.detections))
    )

    return ApiResponse.ok()
