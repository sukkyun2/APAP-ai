import asyncio
from io import BytesIO

from PIL import Image
from fastapi import FastAPI
from fastapi import UploadFile, File

from app.api_response import ApiListResponse
from app.detection import Detection
from app.history import HistorySaveRequest, save_history
from model.detect import detect

app = FastAPI()


def convert(result: dict):
    del result['class']
    return Detection(**result)


@app.post("/detect", response_model=ApiListResponse[Detection])
async def detect_image(file: UploadFile = File(...)) -> ApiListResponse[Detection]:
    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as err:
        return ApiListResponse.bad_request(str(err))

    results = list(map(convert, detect(img)))
    asyncio.create_task(
        save_history(HistorySaveRequest(image=img, detections=results))
    )

    return ApiListResponse[Detection].ok(results)
