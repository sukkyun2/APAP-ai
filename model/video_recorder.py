import time

import cv2

MAX_RECORD_TIME: int = 10
FRAME_SIZES: tuple = (640, 480)
FOURCC: str = 'XVID'
FPS: int = 10

class VideoRecorder:
    def __init__(self):
        self.start_time = None
        self.video_writer = self.initialize_writer()

    def initialize_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*FOURCC)
        return cv2.VideoWriter('output_video.avi', fourcc, FPS, FRAME_SIZES)

    def save_frame(self, frame):
        if self.start_time is None:
            self.start_time = time.time()

        if self.over_max_record_time():
            self.release_video_writer()
            return

        frame = cv2.resize(frame, FRAME_SIZES)
        self.video_writer.write(frame)

    def release_video_writer(self):
        self.video_writer.release()

        self.start_time = None
        self.video_writer = self.initialize_writer()

    def over_max_record_time(self):
        return time.time() - self.start_time > MAX_RECORD_TIME
