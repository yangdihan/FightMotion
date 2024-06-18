import os
import json
from tqdm import tqdm
import numpy as np
import torch
import cv2

from constants import (
    YOLO_THRESHOLD,
    RCNN_THRESHOLD,
    MIN_AREA_RATIO,
    BBOX_DIST_THRESHOLD,
    SKIN_PCT_THRESHOLD,
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

        output_video_path = os.path.join(output_folder, "output_video_mark.mp4")
        out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        for frame in tqdm(self.frames):
            marked_frame = frame.pixels.copy()

            # for contour in frame.contours:
            #     x, y, w, h = map(int, contour.bbox_upper)
            #     cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #     cv2.putText(
            #         marked_frame,
            #         f"Skin: {contour.pct_skin:.2f}",
            #         (x, y - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (255, 0, 0),
            #         2,
            #     )

            #     x, y, w, h = map(int, contour.bbox_lower)
            #     cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     cv2.putText(
            #         marked_frame,
            #         f"Trunk: {contour.trunk_color}",
            #         (x, y - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 255, 0),
            #         2,
            #     )

            #     # Draw contour around the trunk
            #     trunk_contours, _ = cv2.findContours(contour.trunk_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cv2.drawContours(marked_frame, trunk_contours, -1, (0, 255, 255), 2)

            out.write(marked_frame)
            frame_output_path = os.path.join(frame_output_folder, f"{frame.idx}.jpg")
            cv2.imwrite(frame_output_path, marked_frame)

        out.release()

    def direct_connection(self, items, bbox_dist_threshold):
        for frame_idx in range(self.frame_count - 1):
            current_items = getattr(self.frames[frame_idx], items)
            next_items = getattr(self.frames[frame_idx + 1], items)

            for ci in current_items:
                min_dist = float("inf")
                best_match = None
                for ni in next_items:
                    dist = Bbox.bbox_dist(ci, ni)
                    if dist < min_dist and dist <= bbox_dist_threshold:
                        min_dist = dist
                        best_match = ni

                if best_match:
                    ci.next = best_match
                    best_match.prev = ci
        return

    def infer_connection(self, items, bbox_dist_threshold):
        for frame_idx in range(self.frame_count):
            current_items = getattr(self.frames[frame_idx], items)
            for ci in current_items:
                if ci.next is None:
                    for future_frame_idx in range(frame_idx + 1, self.frame_count):
                        future_items = getattr(self.frames[future_frame_idx], items)
                        for fi in future_items:
                            if fi.prev is None:
                                dist_threshold = (
                                    future_frame_idx - frame_idx
                                ) * bbox_dist_threshold
                                if Bbox.bbox_dist(ci, fi) <= dist_threshold:
                                    ci.next = fi
                                    fi.prev = ci
                                    break
                        if ci.next is not None:
                            break

        for frame_idx in range(self.frame_count - 1, 0, -1):
            current_items = getattr(self.frames[frame_idx], items)
            for ci in current_items:
                if ci.prev is None:
                    for past_frame_idx in range(frame_idx - 1, -1, -1):
                        past_items = getattr(self.frames[past_frame_idx], items)
                        for pi in past_items:
                            if pi.next is None:
                                dist_threshold = (
                                    frame_idx - past_frame_idx
                                ) * bbox_dist_threshold
                                if Bbox.bbox_dist(ci, pi) <= dist_threshold:
                                    ci.prev = pi
                                    pi.next = ci
                                    break
                        if ci.prev is not None:
                            break

        return

    def interpolate_bbox(self, start_item, end_item):
        interpolated_items = []
        start_frame = start_item.frame.idx
        end_frame = end_item.frame.idx

        steps = end_frame - start_frame - 1
        if steps <= 0:
            return interpolated_items

        for i in range(1, steps + 1):
            ratio = i / (steps + 1)
            interpolated_xywh = (
                start_item.xywh[0] * (1 - ratio) + end_item.xywh[0] * ratio,
                start_item.xywh[1] * (1 - ratio) + end_item.xywh[1] * ratio,
                start_item.xywh[2] * (1 - ratio) + end_item.xywh[2] * ratio,
                start_item.xywh[3] * (1 - ratio) + end_item.xywh[3] * ratio,
            )
            confidence = (
                start_item.confidence * (1 - ratio) + end_item.confidence * ratio,
            )

            # do not use isinstance when subclass is involved
            if type(start_item) == Bbox:
                interpolated_items.append(
                    Bbox(
                        xywh=interpolated_xywh,
                        confidence=confidence,
                        frame=self.frames[start_frame + i],
                        is_interpolated=True,
                    )
                )
            elif type(start_item) == Contour:
                geometry = Contour.compute_geometry_from_xywh(interpolated_xywh)
                interpolated_items.append(
                    Contour(
                        geometry=geometry,
                        confidence=confidence,
                        frame=self.frames[start_frame + i],
                        is_interpolated=True,
                    )
                )

        return interpolated_items

    def fill_connection(self, items):
        print("Filling missing segmentations...")
        for frame in tqdm(self.frames):
            item_list = getattr(frame, items)
            for item in item_list:
                if item.next and item.next.frame.idx != frame.idx + 1:
                    interpolated_items = self.interpolate_bbox(item, item.next)
                    for ii in interpolated_items:
                        getattr(self.frames[ii.frame.idx], items).append(ii)

                if item.prev and item.prev.frame.idx != frame.idx - 1:
                    interpolated_items = self.interpolate_bbox(item.prev, item)
                    for ii in interpolated_items:
                        getattr(self.frames[ii.frame.idx], items).append(ii)

        return

    def generate_person_bboxes(self):
        min_area = MIN_AREA_RATIO * self.frame_width * self.frame_height

        print(f"Detecting human bbox by YOLO...")
        for frame in tqdm(self.frames):
            frame.extract_fighter_yolo(YOLO_THRESHOLD, min_area)

        self.direct_connection("bboxes", BBOX_DIST_THRESHOLD)
        self.infer_connection("bboxes", BBOX_DIST_THRESHOLD)
        self.fill_connection("bboxes")

        print(f"Masking bbox at each frame...")
        for frame in self.frames:
            mask_bbox = frame.mask_frame_with_bbox(frame.bboxes)
            frame.pixels = frame.crop_frame_with_mask(mask_bbox)

        return

    def generate_fighter_contour(self):
        min_area = MIN_AREA_RATIO * self.frame_width * self.frame_height

        print(f"Detecting fighter contour by RCNN...")
        for frame in tqdm(self.frames):
            frame.extract_fighter_rcnn(RCNN_THRESHOLD, min_area, SKIN_PCT_THRESHOLD)

        self.direct_connection("contours", BBOX_DIST_THRESHOLD)
        self.infer_connection("contours", BBOX_DIST_THRESHOLD)
        self.fill_connection("contours")

        print(f"Masking contour at each frame...")
        for frame in tqdm(self.frames):
            mask_contour = frame.mask_frame_with_contours(frame.contours)
            frame.pixels = frame.crop_frame_with_mask(mask_contour)

        return

    def get_longest_contour_linked_list(self):
        longest_list = []
        visited_hashes = set()

        print("Finding the longest Contour linked-list...")
        for frame in tqdm(self.frames):
            for contour in frame.contours:
                if contour.hash not in visited_hashes:
                    current_list = []
                    current = contour
                    while current and current.hash not in visited_hashes:
                        current_list.append(current)
                        visited_hashes.add(current.hash)
                        current = current.next

                    if len(current_list) > len(longest_list):
                        longest_list = current_list

        return longest_list

    def export_longest_contour_video(self, output_path):
        longest_list = self.get_longest_contour_linked_list()

        output_video_path = os.path.join(output_path, "longest_contour_video.mp4")
        out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        for contour in longest_list:
            frame = contour.frame
            mask = frame.mask_frame_with_contours([contour])
            cropped_frame = frame.crop_frame_with_mask(mask)
            out.write(cropped_frame)

        out.release()
        print(f"Longest contour video saved to {output_video_path}")


def run_extract_fighters(input_video_path, output_folder):
    video_stream = VideoStream(input_video_path)

    video_stream.generate_person_bboxes()
    video_stream.generate_fighter_contour()

    video_stream.cap.release()
    # video_stream.output(output_folder)
    video_stream.export_longest_contour_video(output_folder)
    cv2.destroyAllWindows()
    return
