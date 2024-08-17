from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image as img
from numpy import ndarray
from ultralytics import YOLO

from app.config import settings

model = YOLO(settings.yolo_weight_path)


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
        return int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2)


class DetectionResult:
    def __init__(self, plot_image: np.ndarray, detections: List[Detection]):
        self.plot_image = plot_image
        self.detections = detections

    def get_image(self) -> img:
        return img.fromarray(self.plot_image[..., ::-1])

    def get_encoded_nparr(self) -> ndarray:
        _, encoded_nparr = cv2.imencode('.jpg', self.plot_image)
        return encoded_nparr

    def is_abnormal_pattern_detected(self) -> bool:
        return any(det.class_name == 'person' for det in self.detections)


class DistanceEstimationResult:
    def __init__(self, distance: List[Tuple[int, int, float]], detect_result: DetectionResult):
        self.distance = distance
        self.result = detect_result


tracked_objects: Dict[int, TrackedObject] = {}


def track(image_np: ndarray) -> DetectionResult:
    result = model.track(image_np, persist=True)[0]
    boxes = result.boxes

    class_idxes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.int().cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(class_idxes)
    bboxes = boxes.xyxy.cpu().tolist()

    detections = [Detection(model.names[ci], c, t, bbox) for ci, t, c, bbox in
                  zip(class_idxes, track_ids, confidences, bboxes)]

    return DetectionResult(result.plot(), detections)


def estimate_distance(image_np: ndarray) -> DistanceEstimationResult:
    result = track(image_np)
    non_person_detections = []
    person_detections = []

    for detection in result.detections:
        track_id = detection.track_id

        if track_id not in tracked_objects:
            tracked_objects[track_id] = TrackedObject(track_id, detection.center)
        else:
            tracked_objects[track_id].update_position(detection.center)

        if detection.class_name == 'person':
            person_detections.append(detection)
        else:
            non_person_detections.append(detection)

    # Calculate and return distances between persons and non-persons
    distances = calculate_distance_between_person_and_others(result.plot_image, person_detections,
                                                             non_person_detections)

    return DistanceEstimationResult(distances, result)


def calculate_distance_between_person_and_others(image_np: np.ndarray, person_detections: List[Detection],
                                                 non_person_detections: List[Detection]) -> List[
    Tuple[int, int, float]]:
    """Draw lines and distances between person objects and other objects, and return distances."""

    def calculate_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    distances = []

    for person in person_detections:
        person_center = person.center
        for non_person in non_person_detections:
            other_center = non_person.center
            distance = calculate_distance(person_center, other_center)

            # Draw lines
            cv2.line(image_np, (int(person_center[0]), int(person_center[1])),
                     (int(other_center[0]), int(other_center[1])), (255, 0, 0), 2)

            # Draw distance
            midpoint = ((person_center[0] + other_center[0]) / 2, (person_center[1] + other_center[1]) / 2)
            cv2.putText(image_np, f'{distance:.2f}', (int(midpoint[0]), int(midpoint[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            distances.append((person.track_id, non_person.track_id, distance))

    return distances


# def draw_detections(image_np: np.ndarray, detections: List[Detection]):
#     """Draw detections on the image."""
#     for detection in detections:
#         center = (int(detection.center[0]), int(detection.center[1]))
#         cv2.circle(image_np, center, 5, (0, 255, 0), -1)
#         cv2.putText(image_np, f'ID: {detection.track_id}', (center[0], center[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def detect(image_np: ndarray) -> DetectionResult:
    target_image = img.fromarray(image_np)
    result = model.predict(target_image)[0]

    detections = []
    for box in result.boxes:
        class_name = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()
        detections.append(Detection(class_name, confidence, None, []))

    return DetectionResult(result.plot(), detections)
