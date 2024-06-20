from typing import List, Tuple

import numpy as np
from PIL.Image import Image
from ultralytics import YOLO

from app.config import settings

model = YOLO(settings.yolo_weight_path)


class DetectionResult:
    class_name: str
    confidence: float

    def __init__(self, class_name, confidence):
        self.class_name = class_name
        self.confidence = confidence


def detect(image: Image) -> Tuple[Image, List[dict]]:
    result = model.predict(image)[0]
    predicted_image = Image.fromarray(np.uint8(result.plot(show=False)))

    detections = []
    for box in result.boxes:
        class_name = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()
        detections.append({
            'class_name': class_name,
            'confidence': confidence
        })

    return predicted_image, detections


if __name__ == '__main__':
    img_path = '../tests/resources/bus.jpg'
    img = Image.open(img_path)
    detect(img)
