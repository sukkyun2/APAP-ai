import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def setup_video_output(vid):
    output_path, output_format = '../outputs/output_video.mp4', 'mp4v'
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*output_format)

    return cv2.VideoWriter(output_path, codec, fps, (width, height))


def track(video_path: str):
    vid = cv2.VideoCapture(video_path)
    out = setup_video_output(vid)

    if not vid.isOpened():
        raise Exception('동영상을 열 수 없습니다')

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


if __name__ == '__main__':
    video_path = '../resources/video.avi'
    track(video_path)
