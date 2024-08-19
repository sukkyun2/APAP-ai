import numpy as np
from PIL import Image

from app.history import HistorySaveRequest, save_history
from model.schema import DetectionResult, Detection


def test_save_history():
    given_file_path = 'resources/bus.jpg'
    image = Image.open(given_file_path)
    image_np = np.array(image)

    result = DetectionResult(image_np, [Detection('person', 0.9, 1, []), Detection('banch', 0.8, 1, [])])
    location_name = 'A101'
    req = HistorySaveRequest(result.get_image(), result.detections, location_name)

    save_history(req)
