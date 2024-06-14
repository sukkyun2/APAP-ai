import dataclasses


@dataclasses.dataclass
class Detection:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    name: str
