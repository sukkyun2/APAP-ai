import logging
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as img
from numpy import ndarray
from ultralytics import YOLO

from app.config import settings
from model.schema import Detection, TrackedObject, DetectionResult

model = YOLO(settings.yolo_weight_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model run on the {device}")

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


def estimate_distance(image_np: ndarray) -> tuple[list[tuple[int, int, float]], DetectionResult]:
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

    return distances, result


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


def detect(image_np: ndarray) -> DetectionResult:
    target_image = img.fromarray(image_np)
    result = model.predict(target_image)[0]

    detections = []
    for box in result.boxes:
        class_name = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()
        detections.append(Detection(class_name, confidence, None, []))

    return DetectionResult(result.plot(), detections)
