from enum import StrEnum

from model.detect import estimate_distance, area_intrusion, detect_by_custom_model
from model.schema import DetectionResult


class OperationType(StrEnum):
    ESTIMATE_DISTANCE = "estimate_distance",
    AREA_INTRUSION = "area_intrusion"
    CUSTOM_MODEL = "custom_model"


def define_operation(op: OperationType):
    if op == OperationType.ESTIMATE_DISTANCE:
        return handle_estimate_distance
    elif op == OperationType.AREA_INTRUSION:
        return handle_area_intrusion
    elif op == OperationType.CUSTOM_MODEL:
        return handle_detect_by_custom_model


def handle_estimate_distance(img) -> tuple[bool, DetectionResult]:
    distances, result = estimate_distance(img)
    pattern_detected = any(distance <= 200 for _, _, distance in distances)

    return pattern_detected, result


def handle_area_intrusion(img) -> tuple[bool, DetectionResult]:
    intrusion, result = area_intrusion(img)
    return intrusion, result


def handle_detect_by_custom_model(img) -> tuple[bool, DetectionResult]:
    result = detect_by_custom_model(img)
    return len(result.detections) > 0, result
