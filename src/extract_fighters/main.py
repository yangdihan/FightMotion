import os
import time
import json
from tqdm import tqdm
import numpy as np
import torch
import cv2

from extract_fighters.constants import (
    MIN_AREA_RATIO,
    YOLO_THRESHOLD,
    BBOX_DIST_THRESHOLD,
    RCNN_THRESHOLD,
    SIGNIFICANT_DROP_RATIO,
)
from extract_fighters.extract_yolo import main as run_yolo_bbox
from extract_fighters.extract_rcnn import main as run_rcnn_contour


class Frame:
    def __init__(self) -> None:
        self.idx = None
        self.bbox = []
        self.contour = []
        self.bbox_mask = None
        self.contour_mask = None
        self.frame = None
        return


class VideoStream:
    def __init__(self, input_video_path) -> None:
        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.get_video_properties()

        self.frames = [
            Frame() for _ in range(self.frame_count)
        ]  # Create unique Frame objects

        return

    def get_video_properties(self):
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return

    def read_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
        ret, frame = self.cap.read()
        return ret, frame

    def output(self, output_folder):
        print(f"Exporting frames...")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.frame_output_folder = os.path.join(output_folder, "frames")
        if not os.path.exists(self.frame_output_folder):
            os.makedirs(self.frame_output_folder)

        output_video_path = os.path.join(output_folder, "output_video.mp4")
        self.out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        for frame in tqdm(self.frames):
            # cv2.imshow("debug", frame.frame)
            self.out.write(frame.frame)
            frame_output_path = os.path.join(
                self.frame_output_folder, f"{frame.idx}.jpg"
            )
            cv2.imwrite(frame_output_path, frame.frame)

        self.out.release()
        return


def run_extract_fighters(input_video_path, output_folder):

    video_stream = VideoStream(input_video_path)

    video_stream = run_yolo_bbox(
        video_stream, YOLO_THRESHOLD, MIN_AREA_RATIO, BBOX_DIST_THRESHOLD
    )

    video_stream = run_rcnn_contour(
        video_stream, RCNN_THRESHOLD, SIGNIFICANT_DROP_RATIO, BBOX_DIST_THRESHOLD
    )

    video_stream.cap.release()

    video_stream.output(output_folder)

    cv2.destroyAllWindows()
    return
