import os
import json
from tqdm import tqdm
import numpy as np
import torch
import cv2

from constants import (
    DEVICE,
    MASK_EXPAND_RATIO,
    YOLO_THRESHOLD,
    MIN_AREA_RATIO,
    BBOX_DIST_THRESHOLD,
    RCNN_THRESHOLD,
    SIGNIFICANT_DROP_RATIO,
)
from frame import Frame
from bbox import Bbox
from contour import Contour


class VideoStream:
    def __init__(self, input_video_path) -> None:
        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = [None] * self.frame_count

        for frame_idx in range(self.frame_count):
            ret, pixels = self.read_frame(frame_idx)
            if not ret:
                break
            self.frames[frame_idx] = Frame(frame_idx, pixels)
        return

    def read_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return ret, frame

    def output(self, output_folder):
        print(f"Exporting frames...")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        frame_output_folder = os.path.join(output_folder, "frames")
        if not os.path.exists(frame_output_folder):
            os.makedirs(frame_output_folder)

        output_video_path = os.path.join(output_folder, "output_video.mp4")
        out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        for frame in tqdm(self.frames):
            out.write(frame.pixels)
            frame_output_path = os.path.join(frame_output_folder, f"{frame.idx}.jpg")
            cv2.imwrite(frame_output_path, frame.pixels)

        out.release()

    def direct_connection(self, bbox_dist_threshold):

        for frame_idx in range(self.frame_count - 1):

            current_bboxes = self.frames[frame_idx].bboxes
            next_bboxes = self.frames[frame_idx + 1].bboxes

            for cb in current_bboxes:
                min_dist = float("inf")
                best_match = None
                for nb in next_bboxes:
                    dist = Bbox.bbox_dist(cb, nb)
                    if dist < min_dist and dist <= bbox_dist_threshold:
                        min_dist = dist
                        best_match = nb

                if best_match:
                    cb.next = best_match
                    best_match.prev = cb
        return

    def infer_connection(self, bbox_dist_threshold):
        # Forward iteration: find the next linked bbox for bboxes without next
        for frame_idx in range(self.frame_count):
            current_bboxes = self.frames[frame_idx].bboxes
            for cb in current_bboxes:
                if cb.next is None:
                    for future_frame_idx in range(frame_idx + 1, self.frame_count):
                        future_bboxes = self.frames[future_frame_idx].bboxes
                        for fb in future_bboxes:
                            if fb.prev is None:
                                dist_threshold = (
                                    future_frame_idx - frame_idx
                                ) * bbox_dist_threshold
                                if Bbox.bbox_dist(cb, fb) <= dist_threshold:
                                    cb.next = fb
                                    fb.prev = cb
                                    break
                        if cb.next is not None:
                            break

        # Backward iteration: find the previous linked bbox for bboxes without prev
        for frame_idx in range(self.frame_count - 1, 0, -1):
            current_bboxes = self.frames[frame_idx].bboxes
            for cb in current_bboxes:
                if cb.prev is None:
                    for past_frame_idx in range(frame_idx - 1, -1, -1):
                        past_bboxes = self.frames[past_frame_idx].bboxes
                        for pb in past_bboxes:
                            if pb.next is None:
                                dist_threshold = (
                                    frame_idx - past_frame_idx
                                ) * bbox_dist_threshold
                                if Bbox.bbox_dist(cb, pb) <= dist_threshold:
                                    cb.prev = pb
                                    pb.next = cb
                                    break
                        if cb.prev is not None:
                            break

        return

    def interpolate_bbox(self, start_bbox, end_bbox):
        interpolated_bboxes = []
        start_frame = start_bbox.frame.idx
        end_frame = end_bbox.frame.idx

        steps = end_frame - start_frame - 1
        if steps <= 0:
            return interpolated_bboxes

        for i in range(1, steps + 1):
            ratio = i / (steps + 1)
            interpolated_bbox = (
                start_bbox.xywh[0] * (1 - ratio) + end_bbox.xywh[0] * ratio,
                start_bbox.xywh[1] * (1 - ratio) + end_bbox.xywh[1] * ratio,
                start_bbox.xywh[2] * (1 - ratio) + end_bbox.xywh[2] * ratio,
                start_bbox.xywh[3] * (1 - ratio) + end_bbox.xywh[3] * ratio,
            )
            confidence = (
                start_bbox.confidence * (1 - ratio) + end_bbox.confidence * ratio,
            )
            interpolated_bboxes.append(
                Bbox(
                    xywh=interpolated_bbox,
                    frame=self.frames[start_frame + i],
                    confidence=confidence,
                    is_interpolated=True,
                )
            )

        return interpolated_bboxes

    def fill_connection(self):
        for frame in self.frames:
            for bbox in frame.bboxes:
                if bbox.next and bbox.next.frame.idx != frame.idx + 1:
                    interpolated_bboxes = self.interpolate_bbox(bbox, bbox.next)
                    for ib in interpolated_bboxes:
                        self.frames[ib.frame.idx].bboxes.append(ib)

                if bbox.prev and bbox.prev.frame.idx != frame.idx - 1:
                    interpolated_bboxes = self.interpolate_bbox(bbox.prev, bbox)
                    for ib in interpolated_bboxes:
                        self.frames[ib.frame.idx].bboxes.append(ib)

        return

    def generate_person_bboxes(self):
        min_area = MIN_AREA_RATIO * self.frame_width * self.frame_height

        print(f"Detecting human bbox by YOLO...")
        for frame in tqdm(self.frames):
            frame.extract_person_yolo(min_area)

        self.direct_connection(BBOX_DIST_THRESHOLD)
        self.infer_connection(BBOX_DIST_THRESHOLD)
        self.fill_connection()

        print(f"Masking bbox at each frame...")
        for frame in self.frames:
            mask_bbox = frame.mask_frame_with_bbox(frame.bboxes)
            frame.pixels = frame.crop_frame_with_mask(mask_bbox)

        return

    def generate_fighter_contour(self):
        print(f"Detecting fighter contour by RCNN...")

        previous_non_blank_pixel_count = None
        top_contours_last = []

        print(f"Masking contour at each frame...")
        for frame in tqdm(self.frames):

            contours_top2 = frame.extract_fighter_contour()

            mask_contour2 = frame.mask_frame_with_contours(contours_top2)

            non_blank_pixel_count = cv2.countNonZero(mask_contour2)

            if Contour.significant_drop(
                previous_non_blank_pixel_count,
                non_blank_pixel_count,
                SIGNIFICANT_DROP_RATIO,
            ):
                contours_top2 = Contour.infer_missing_contours(
                    top_contours_last, contours_top2, BBOX_DIST_THRESHOLD
                )
                mask_contour2 = frame.mask_frame_with_contours(contours_top2)

            frame.contours = contours_top2
            frame.mask_contour = mask_contour2
            frame.pixels = frame.crop_frame_with_mask(mask_contour2)

            previous_non_blank_pixel_count = non_blank_pixel_count
            top_contours_last = contours_top2

        return


def run_extract_fighters(input_video_path, output_folder):
    video_stream = VideoStream(input_video_path)

    video_stream.generate_person_bboxes()
    video_stream.generate_fighter_contour()

    video_stream.cap.release()
    video_stream.output(output_folder)
    cv2.destroyAllWindows()
    return
