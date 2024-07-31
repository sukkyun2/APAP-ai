import dataclasses
from typing import List, Optional

import cv2
from PIL import Image as img
from PIL.Image import Image
from numpy import ndarray
from ultralytics import YOLO

from app.config import settings

model = YOLO(settings.yolo_weight_path)


@dataclasses.dataclass
class Detection:
    class_name: str
    confidence: float
    track_id: Optional[int]


@dataclasses.dataclass
class DetectionResult:
    predict_image_np: ndarray
    detections: List[Detection]

    def get_image(self) -> Image:
        return img.fromarray(self.predict_image_np[..., ::-1])

    def get_encoded_nparr(self):
        _, predicted_encode_nparr = cv2.imencode('.jpg', self.predict_image_np)
        return predicted_encode_nparr


def track(image_np: ndarray) -> DetectionResult:
    result = model.track(image_np, persist=True)[0]
    boxes = result.boxes

    class_idxes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.int().cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id else [None] * len(class_idxes)

    detections = [Detection(model.names[ci], c, t) for ci, t, c in zip(class_idxes, track_ids, confidences)]

    return DetectionResult(result.plot(), detections)


def detect(image_np: ndarray) -> DetectionResult:
    target_image = img.fromarray(image_np)
    result = model.predict(target_image)[0]

    detections = []
    for box in result.boxes:
        class_name = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()
        detections.append(Detection(class_name, confidence, None))

    return DetectionResult(result.plot(), detections)
