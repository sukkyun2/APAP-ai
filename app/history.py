import base64
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import List

import httpx
from PIL.Image import Image
from pydantic import BaseModel

from app.config import settings
from app.detection import Detection


class HistorySaveRequest(BaseModel):
    datetime: str
    label: str
    image: str  # encoded base64 string

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

        super().__init__(datetime=datetime.now().isoformat(), label=summary_detections(), image=convert_base64())


async def save_history(req: HistorySaveRequest):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{settings.history_api}/info", data=req.dict())
    except httpx.RequestError as exc:
        print(str(exc))
