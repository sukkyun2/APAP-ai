import asyncio
import base64
import logging
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import List

import httpx
from PIL.Image import Image
from pydantic import BaseModel

from app.config import settings
from model.detect import Detection, DetectionResult
from model.operations import OperationType

logging.basicConfig(level=logging.INFO)


class HistorySaveRequest(BaseModel):
    localDateTime: str
    label: str
    base64Image: str
    cameraName: str

    def __init__(self, image: Image, detections: List[Detection], location_name: str, op: OperationType):
        def convert_base64():
            image_format = 'JPEG'
            buffered = BytesIO()
            image.save(buffered, format=image_format)

            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            mime_type = f"image/{image_format.lower()}"

            return f"data:{mime_type};base64,{base64_str}"

        def summary_detections():
            counter = Counter(list(map(lambda d: d.class_name, detections)))
            label = ' '.join(f'{k} {v}' for k, v in counter.items())

            return f"{op.value}:{label}"

        super().__init__(localDateTime=datetime.now().isoformat(), label=summary_detections(),
                         base64Image=convert_base64(), cameraName=location_name)


async def async_save_history(result: DetectionResult, location_name: str, op: OperationType):
    logging.info("이상상황이 발생하여 이력을 저장합니다")

    asyncio.create_task(
        save_history(HistorySaveRequest(result.get_image(), result.detections, location_name, op))
    )


async def save_history(req: HistorySaveRequest):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{settings.history_api}/api/infos", json=req.dict())
    except httpx.RequestError as exc:
        print(str(exc))
