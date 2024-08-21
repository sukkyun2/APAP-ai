from enum import StrEnum

from model.detect import estimate_distance, area_intrusion
from model.schema import DetectionResult


class OperationType(StrEnum):
    ESTIMATE_DISTANCE = "estimate_distance",
    AREA_INTRUSION = "area_intrusion"


def define_operation(op: OperationType):
    if op == OperationType.ESTIMATE_DISTANCE:
        return handle_estimate_distance
    elif op == OperationType.AREA_INTRUSION:
        return handle_area_intrusion


def handle_estimate_distance(img) -> tuple[bool, DetectionResult]:
    distances, result = estimate_distance(img)
    pattern_detected = any(distance <= 200 for _, _, distance in distances)

    return pattern_detected, result


def handle_area_intrusion(img) -> tuple[bool, DetectionResult]:
    intrusion, result = area_intrusion(img)
    return intrusion, result
