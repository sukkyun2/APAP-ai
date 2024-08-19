from typing import List, Optional

import numpy as np
from PIL import Image as img


class TrackedObject:
    def __init__(self, obj_id: int, center: tuple[int, int]):
        self.obj_id = obj_id
        self.current_center = center

    def update_position(self, new_center: tuple[int, int]):
        self.current_center = new_center


class Detection:
    def __init__(self, class_name: str, confidence: float, track_id: Optional[int], bbox: List[float]):
        self.class_name = class_name
        self.confidence = confidence
        self.track_id = track_id
        self.bbox = bbox
        self.center = self.calculate_centroid(bbox)

    @staticmethod
    def calculate_centroid(bbox: List[float]) -> tuple[int, int]:
        """Calculates the centroid of a bounding box."""
        if not bbox:
            return -1, -1

        return int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2)


class DetectionResult:
    def __init__(self, plot_image: np.ndarray, detections: List[Detection]):
        self.plot_image = plot_image
        self.detections = detections

    def get_image(self) -> img:
        return img.fromarray(self.plot_image[..., ::-1])
