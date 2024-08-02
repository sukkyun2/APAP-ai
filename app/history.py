import asyncio
import base64
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import List

import httpx
from PIL.Image import Image
from pydantic import BaseModel

from app.config import settings
from model.detect import Detection, DetectionResult


class HistorySaveRequest(BaseModel):
    localDateTime: str
    label: str
    base64Image: str

    def __init__(self, image: Image, detections: List[Detection]):
        def convert_base64():
            image_format = 'JPEG'
            buffered = BytesIO()
            image.save(buffered, format=image_format)

            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            mime_type = f"image/{image_format.lower()}"

            return f"data:{mime_type};base64,{base64_str}"

        def summary_detections():
            counter = Counter(list(map(lambda d: d.class_name, detections)))
            return ' '.join(f'{k} {v}' for k, v in counter.items())

        super().__init__(localDateTime=datetime.now().isoformat(), label=summary_detections(), base64Image=convert_base64())


async def async_save_history(result: DetectionResult):
    asyncio.create_task(
        save_history(HistorySaveRequest(image=result.get_image(), detections=result.detections))
    )


async def save_history(req: HistorySaveRequest):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{settings.history_api}/api/infos", json=req.dict())
    except httpx.RequestError as exc:
        print(str(exc))
