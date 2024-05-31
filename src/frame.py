import numpy as np
import torch
import cv2

from constants import MASK_EXPAND_RATIO, YOLO_MODEL
from bbox import Bbox


class Frame:
    def __init__(self, idx) -> None:
        self.idx = idx
        self.bboxes = []
        self.mask_bbox = None
        self.contours = []
        self.mask_contour = None
        self.frame = None

    @staticmethod
    def extract_person_yolo(frame, yolo_model, threshold, min_area):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_model(img)
        bboxes = []

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if cls == 0 and conf >= threshold:
                x1, y1, x2, y2 = map(int, box)
                box_area = (x2 - x1) * (y2 - y1)
                if box_area >= min_area:
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    if bbox not in bboxes:  # Ensure no duplicates
                        bboxes.append(bbox)

        return bboxes

    @staticmethod
    def get_masked_frame(frame, bboxes, frame_width, frame_height):
        mask = np.zeros_like(frame)
        for bbox in bboxes:
            x, y, w, h = map(int, bbox.bbox)
            x, y, w, h = Bbox.expand_bbox(
                x, y, w, h, frame_width, frame_height, MASK_EXPAND_RATIO
            )
            mask[y : y + h, x : x + w] = frame[y : y + h, x : x + w]
        return mask

    @staticmethod
    def run_yolo_bbox(
        video_stream, yolo_threshold, min_area_ratio, bbox_dist_threshold
    ):
        min_area = min_area_ratio * video_stream.frame_width * video_stream.frame_height

        detections = video_stream.get_yolo_detections(
            YOLO_MODEL, yolo_threshold, min_area
        )
        print(f"Interpolating human bbox...")
        linked_bboxes = video_stream.direct_connection(detections, bbox_dist_threshold)
        linked_bboxes = video_stream.infer_connection(
            linked_bboxes, bbox_dist_threshold
        )
        linked_bboxes = video_stream.interpolate_missing_bboxes(linked_bboxes)

        for frame_idx in range(video_stream.frame_count):
            ret, frame_data = video_stream.read_frame(frame_idx)
            if not ret:
                break

            mask_bbox = Frame.get_masked_frame(
                frame_data,
                linked_bboxes.get(frame_idx, []),
                video_stream.frame_width,
                video_stream.frame_height,
            )
            video_stream.frames[frame_idx].bboxes = linked_bboxes.get(frame_idx, [])
            video_stream.frames[frame_idx].mask_bbox = mask_bbox

        return video_stream
