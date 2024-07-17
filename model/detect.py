import dataclasses
from enum import Enum
from typing import List

import cv2
from PIL import Image as img
from PIL.Image import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from numpy import ndarray
from ultralytics import YOLO

from app.config import settings

model = YOLO(settings.yolo_weight_path)
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)


@dataclasses.dataclass
class Detection:
    class_name: str
    confidence: float
    track_id: int


@dataclasses.dataclass
class DetectionResult:
    predict_image_np: ndarray
    detections: List[Detection]

    def get_image(self) -> Image:
        return img.fromarray(self.predict_image_np)


class BBoxColor(Enum):
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)


def detect(image_np: ndarray) -> DetectionResult:
    result = model.predict(image_np)[0]

    detection_records = []
    for data in result.boxes.data.tolist():
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        confidence, label = float(data[4]), int(data[5])

        detection_records.append([[xmin, ymin, xmax, ymax], confidence, label])

    tracks = tracker.update_tracks(detection_records, frame=image_np)

    detections = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id, class_name, confidence = track.track_id, model.names[track.det_class], track.det_conf
        ltrb = track.to_ltrb()

        if not class_name or not confidence:
            continue

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), BBoxColor.GREEN.value, 2)
        # cv2.rectangle(image_np, (xmin, ymin - 20), (xmin + 20, ymin), BBoxColor.GREEN.value, -1)

        label = f"{class_name} {track_id}: {confidence:.2f}"
        cv2.rectangle(image_np, (xmin, ymin - 30), (xmax, ymin), BBoxColor.GREEN.value, -1)
        cv2.putText(image_np, label, (xmin, ymin - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBoxColor.WHITE.value, 2)

        detections.append(Detection(class_name, confidence, track_id))

    return DetectionResult(image_np, detections)
