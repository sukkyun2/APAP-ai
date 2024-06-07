from typing import List

import cv2
import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def detect(image: Image) -> List[dict]:
    results = model(image)
    return results.pandas().xyxy[0].to_dict(orient='records')


def load_image(image_path: str):
    return Image.open(image_path)


def visualize_results(results):
    results.render()
    img_with_boxes = results.ims[0]

    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 Inference', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = 'https://ultralytics.com/images/bus.jpg'
    img = load_image(img_path)
    results = model(img)
    visualize_results(results)
