import logging
import time

import cv2
from numpy import ndarray

MAX_RECORD_TIME: int = 30


class VideoRecorder:
    def __init__(self):
        self.frame_buffer = []
        self.start_time = None
        self.is_recording = False
        self.duration = 30  # Seconds

    def start_record_if_not(self):
        if self.is_recording:
            return

        logging.info("start record")

        self.start_time = time.time()
        self.is_recording = True
        self.frame_buffer = []  # Reset buffer for new recording

    def record_frame(self, nparr: ndarray):
        self.frame_buffer.append(nparr)

        if self.over_max_record_time():
            self.save_video()
            self.is_recording = False
            self.frame_buffer = []

    def save_video(self):
        frames = self.frame_buffer

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_{}.mp4'.format(time.strftime("%Y%m%d_%H%M%S")), fourcc, 20.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

        logging.info("video saved!")

    def over_max_record_time(self):
        return time.time() - self.start_time > MAX_RECORD_TIME
