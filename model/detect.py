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


def define_zone(image_np: ndarray) -> list[int]:
    height, width = image_np.shape[:2]
    return [width // 4, height // 4, width // 2, height // 2]


def draw_zone(image_np: ndarray, zone: list[int]) -> ndarray:
    image_with_zone = cv2.rectangle(image_np.copy(), (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 2)
    label_text = "Danger Zone"
    label_position = (zone[0] + 10, zone[1] - 10)
    cv2.putText(image_with_zone, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image_with_zone


def check_intruded(bbox: list[float], zone: list[int]) -> bool:
    x1, y1, x2, y2 = map(int, bbox)
    return not (x2 < zone[0] or x1 > zone[2] or y2 < zone[1] or y1 > zone[3])


def area_intrusion(image_np: ndarray) -> tuple[bool, DetectionResult]:
    zone = define_zone(image_np)
    result = track(image_np)

    image_with_zone = draw_zone(image_np, zone)
    intrusion_detections = []
    intrusion = False

    for d in result.detections:
        class_name, track_id, confidence, bbox = d.class_name, d.track_id, d.confidence, d.bbox
        x1, y1, x2, y2 = map(int, bbox)

        if class_name == 'person' and check_intruded(bbox, zone):
            intrusion = True
            intrusion_detections.append(d)

            image_with_zone = cv2.rectangle(image_with_zone, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_zone, f'{class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return intrusion, DetectionResult(image_with_zone, intrusion_detections)


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
