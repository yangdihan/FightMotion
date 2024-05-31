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
        self.frames = [Frame(idx) for idx in range(self.frame_count)]

    def read_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
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
            out.write(frame.frame)
            frame_output_path = os.path.join(frame_output_folder, f"{frame.idx}.jpg")
            cv2.imwrite(frame_output_path, frame.frame)

        out.release()

    def get_yolo_detections(self, yolo_model, threshold, min_area):
        print(f"Detecting human bbox by YOLO...")
        detections = {}
        for frame_idx in tqdm(range(self.frame_count)):
            ret, frame_data = self.read_frame(frame_idx)
            if not ret:
                break

            bboxes = Frame.extract_person_yolo(
                frame_data, yolo_model, threshold, min_area
            )
            detections[frame_idx] = bboxes

        return detections

    def direct_connection(self, detections, bbox_dist_threshold):
        linked_bboxes = {}

        for frame_idx, bboxes in detections.items():
            linked_bboxes[frame_idx] = [
                Bbox(bbox=bbox, frame_idx=frame_idx) for bbox in bboxes
            ]

        for frame_idx in range(len(detections) - 1):
            current_bboxes = linked_bboxes[frame_idx]
            next_bboxes = linked_bboxes[frame_idx + 1]

            for cb in current_bboxes:
                min_dist = float("inf")
                best_match = None
                for nb in next_bboxes:
                    dist = Bbox.bbox_dist(cb.bbox, nb.bbox)
                    if dist < min_dist and dist <= bbox_dist_threshold:
                        min_dist = dist
                        best_match = nb

                if best_match:
                    cb.next = best_match
                    best_match.prev = cb

        return linked_bboxes

    def infer_connection(self, linked_bboxes, bbox_dist_threshold):
        # Forward iteration: find the next linked bbox for bboxes without next
        for frame_idx in range(len(linked_bboxes) - 1):
            current_bboxes = linked_bboxes[frame_idx]
            for cb in current_bboxes:
                if cb.next is None:
                    for future_frame_idx in range(frame_idx + 1, len(linked_bboxes)):
                        future_bboxes = linked_bboxes[future_frame_idx]
                        for fb in future_bboxes:
                            if fb.prev is None:
                                dist_threshold = (
                                    future_frame_idx - frame_idx
                                ) * bbox_dist_threshold
                                if Bbox.bbox_dist(cb.bbox, fb.bbox) <= dist_threshold:
                                    cb.next = fb
                                    fb.prev = cb
                                    break
                        if cb.next is not None:
                            break

        # Backward iteration: find the previous linked bbox for bboxes without prev
        for frame_idx in range(len(linked_bboxes) - 1, 0, -1):
            current_bboxes = linked_bboxes[frame_idx]
            for cb in current_bboxes:
                if cb.prev is None:
                    for past_frame_idx in range(frame_idx - 1, -1, -1):
                        past_bboxes = linked_bboxes[past_frame_idx]
                        for pb in past_bboxes:
                            if pb.next is None:
                                dist_threshold = (
                                    frame_idx - past_frame_idx
                                ) * bbox_dist_threshold
                                if Bbox.bbox_dist(cb.bbox, pb.bbox) <= dist_threshold:
                                    cb.prev = pb
                                    pb.next = cb
                                    break
                        if cb.prev is not None:
                            break

        return linked_bboxes

    def interpolate_missing_bboxes(self, linked_bboxes):
        all_bboxes = []
        for frame_idx in sorted(linked_bboxes.keys()):
            all_bboxes.extend(linked_bboxes[frame_idx])

        for bbox in all_bboxes:
            if bbox.next and bbox.next.frame_idx != bbox.frame_idx + 1:
                interpolated_bboxes = Bbox.interpolate_bbox(bbox, bbox.next)
                for ib in interpolated_bboxes:
                    if ib.frame_idx not in linked_bboxes:
                        linked_bboxes[ib.frame_idx] = []
                    linked_bboxes[ib.frame_idx].append(ib)

            if bbox.prev and bbox.prev.frame_idx != bbox.frame_idx - 1:
                interpolated_bboxes = Bbox.interpolate_bbox(bbox.prev, bbox)
                for ib in interpolated_bboxes:
                    if ib.frame_idx not in linked_bboxes:
                        linked_bboxes[ib.frame_idx] = []
                    linked_bboxes[ib.frame_idx].append(ib)

        return linked_bboxes

    def infer_missing_contours(self, contours_last, contours_this, bbox_dist_threshold):
        # Get bounding boxes for last and current contours
        bboxes_last = [cv2.boundingRect(contour.geometry) for contour in contours_last]
        bboxes_this = [cv2.boundingRect(contour.geometry) for contour in contours_this]

        paired_last = [False] * len(contours_last)
        paired_this = [False] * len(contours_this)

        for i, bbox_this in enumerate(bboxes_this):
            for j, bbox_last in enumerate(bboxes_last):
                if Bbox.bbox_dist(bbox_this, bbox_last) <= bbox_dist_threshold:
                    paired_this[i] = True
                    paired_last[j] = True

        # Add unpaired contours from the last frame to the current frame
        for i, paired in enumerate(paired_last):
            if not paired:
                contours_this.append(contours_last[i])

        return contours_this


def run_extract_fighters(input_video_path, output_folder):
    video_stream = VideoStream(input_video_path)

    video_stream = Frame.run_yolo_bbox(
        video_stream, YOLO_THRESHOLD, MIN_AREA_RATIO, BBOX_DIST_THRESHOLD
    )

    video_stream = Contour.run_rcnn_contour(
        video_stream, RCNN_THRESHOLD, SIGNIFICANT_DROP_RATIO, BBOX_DIST_THRESHOLD
    )

    video_stream.cap.release()
    video_stream.output(output_folder)
    cv2.destroyAllWindows()
    return
