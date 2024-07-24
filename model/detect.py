import dataclasses
from typing import List, Optional

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
        return img.fromarray(self.predict_image_np)


def track(image_np: ndarray) -> DetectionResult:
    result = model.track(image_np, persist=True)[0]

    # boxes = result.boxes.xywh.cpu()
    class_idxes = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.int().cpu().tolist()
    track_ids = result.boxes.id.int().cpu().tolist()

    detections = [Detection(model.names[ci], c, t) for ci, t, c in zip(class_idxes, track_ids, confidences)]

    return DetectionResult(result.plot(), detections)


def detect(image_np: ndarray) -> DetectionResult:
    result = model.predict(image_np)[0]

    detections = []
    for box in result.boxes:
        class_name = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()
        detections.append(Detection(class_name, confidence, None))

    return DetectionResult(image_np, detections)
