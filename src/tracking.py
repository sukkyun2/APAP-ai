import cv2
import torch


def setup_video_output(vid):
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*output_format)

    return cv2.VideoWriter(output_path, codec, fps, (width, height))


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    video_path = '../resources/video.avi'
    output_path = '../outputs/output_video.mp4'
    output_format = 'mp4v'

    vid = cv2.VideoCapture(video_path)
    out = setup_video_output(vid)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        results = model(frame)

        results.render()
        frame = results.ims[0]

        out.write(frame)

        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
