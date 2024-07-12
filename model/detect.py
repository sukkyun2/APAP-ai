import dataclasses
from typing import List

import numpy as np
from PIL import Image as img
from PIL.Image import Image
from ultralytics import YOLO

from app.config import settings

model = YOLO(settings.yolo_weight_path)


@dataclasses.dataclass
class Detection:
    class_name: str
    confidence: float


@dataclasses.dataclass
class DetectionResult:
    predicted_image: Image
    detections: List[Detection]

    def get_image_to_nparray(self):
        return np.array(self.predicted_image)


def detect(image: Image) -> DetectionResult:
    result = model.predict(image)[0]
    predicted_image = img.fromarray(np.uint8(result.plot(show=False)))

    detections = []
    for box in result.boxes:
        class_name = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()
        detections.append(Detection(class_name, confidence))

    return DetectionResult(predicted_image, detections)
