import cv2
import torch
from PIL import Image


def load_image(image_path):
    return Image.open(image_path)


def visualize_results(results):
    results.render()
    img_with_boxes = results.ims[0]

    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 Inference', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    image_path = '../resources/image.jpg'
    img = load_image(image_path)
    results = model(img)
    visualize_results(results)
