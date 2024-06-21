import dataclasses


@dataclasses.dataclass
class Detection:
    class_name: str
    confidence: float
