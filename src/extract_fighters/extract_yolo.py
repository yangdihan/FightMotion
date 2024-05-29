from tqdm import tqdm
import numpy as np
import torch
import cv2

from extract_fighters.constants import DEVICE, MASK_EXPAND_RATIO
from extract_fighters.utils import LinkedBbox, bbox_dist, read_frame

# Load YOLOv5 model
YOLO_MODEL = (
    torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval().to(DEVICE)
)
YOLO_MODEL.classes = [0]  # Set model to detect only people (class 0)


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


def get_yolo_detections(video_stream, yolo_model, threshold, min_area):
    print(f"Detecting human bbox by YOLO...")
    detections = {}
    for frame_idx in tqdm(range(video_stream.frame_count)):
        ret, frame = read_frame(video_stream.cap, frame_idx)
        if not ret:
            break

        bboxes = extract_person_yolo(frame, yolo_model, threshold, min_area)
        detections[frame_idx] = bboxes

    return detections


def direct_connection(detections, bbox_dist_threshold):
    linked_bboxes = {}

    for frame_idx, bboxes in detections.items():
        linked_bboxes[frame_idx] = [
            LinkedBbox(bbox=bbox, frame=frame_idx) for bbox in bboxes
        ]

    for frame_idx in range(len(detections) - 1):
        current_bboxes = linked_bboxes[frame_idx]
        next_bboxes = linked_bboxes[frame_idx + 1]

        for cb in current_bboxes:
            min_dist = float("inf")
            best_match = None
            for nb in next_bboxes:
                dist = bbox_dist(cb.bbox, nb.bbox)
                if dist < min_dist and dist <= bbox_dist_threshold:
                    min_dist = dist
                    best_match = nb

            if best_match:
                cb.next = best_match
                best_match.prev = cb

    return linked_bboxes


def infer_connection(linked_bboxes, bbox_dist_threshold):
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
                            if bbox_dist(cb.bbox, fb.bbox) <= dist_threshold:
                                # print(
                                #     f"{frame_idx}:{cb.hash}->{future_frame_idx}:{fb.hash}={bbox_dist(cb.bbox, fb.bbox)}<{dist_threshold}"
                                # )
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
                            if bbox_dist(cb.bbox, pb.bbox) <= dist_threshold:
                                # print(
                                #     f"{frame_idx}:{cb.hash}->{past_frame_idx}:{pb.hash}={bbox_dist(cb.bbox, pb.bbox)}<{dist_threshold}"
                                # )
                                cb.prev = pb
                                pb.next = cb
                                break
                    if cb.prev is not None:
                        break

    return linked_bboxes


def process_detections(detections, bbox_dist_threshold):
    linked_bboxes = direct_connection(detections, bbox_dist_threshold)
    linked_bboxes = infer_connection(linked_bboxes, bbox_dist_threshold)
    return linked_bboxes


def interpolate_bbox(start_bbox, end_bbox):
    interpolated_bboxes = []
    start_frame = start_bbox.frame
    end_frame = end_bbox.frame

    steps = end_frame - start_frame - 1
    if steps <= 0:
        return interpolated_bboxes

    for i in range(1, steps + 1):
        ratio = i / (steps + 1)
        interpolated_bbox = (
            start_bbox.bbox[0] * (1 - ratio) + end_bbox.bbox[0] * ratio,
            start_bbox.bbox[1] * (1 - ratio) + end_bbox.bbox[1] * ratio,
            start_bbox.bbox[2] * (1 - ratio) + end_bbox.bbox[2] * ratio,
            start_bbox.bbox[3] * (1 - ratio) + end_bbox.bbox[3] * ratio,
        )
        interpolated_bboxes.append(
            LinkedBbox(
                bbox=interpolated_bbox, frame=start_frame + i, is_interpolated=True
            )
        )

    return interpolated_bboxes


def interpolate_missing_bboxes(linked_bboxes):
    all_bboxes = []
    for frame_idx in sorted(linked_bboxes.keys()):
        all_bboxes.extend(linked_bboxes[frame_idx])

    for bbox in all_bboxes:
        if bbox.next and bbox.next.frame != bbox.frame + 1:
            interpolated_bboxes = interpolate_bbox(bbox, bbox.next)
            for ib in interpolated_bboxes:
                if ib.frame not in linked_bboxes:
                    linked_bboxes[ib.frame] = []
                linked_bboxes[ib.frame].append(ib)

        if bbox.prev and bbox.prev.frame != bbox.frame - 1:
            interpolated_bboxes = interpolate_bbox(bbox.prev, bbox)
            for ib in interpolated_bboxes:
                if ib.frame not in linked_bboxes:
                    linked_bboxes[ib.frame] = []
                linked_bboxes[ib.frame].append(ib)

    return linked_bboxes


def expand_bbox(x, y, w, h, frame_width, frame_height, expand_ratio):
    x -= int(expand_ratio * w)
    y -= int(expand_ratio * h)
    w = int(1 + 2 * expand_ratio) * w
    h = int(1 + 2 * expand_ratio) * h
    x = max(0, x)
    y = max(0, y)
    w = min(frame_width - x, w)
    h = min(frame_height - y, h)
    return x, y, w, h


def get_masked_frame(frame, bboxes, frame_width, frame_height):
    mask = np.zeros_like(frame)
    for node in bboxes:
        x, y, w, h = map(int, node.bbox)
        x, y, w, h = expand_bbox(
            x, y, w, h, frame_width, frame_height, MASK_EXPAND_RATIO
        )
        mask[y : y + h, x : x + w] = frame[y : y + h, x : x + w]
    return mask


def main(video_stream, yolo_threshold, min_area_ratio, bbox_dist_threshold):

    min_area = min_area_ratio * video_stream.frame_width * video_stream.frame_height

    detections = get_yolo_detections(video_stream, YOLO_MODEL, yolo_threshold, min_area)
    print(f"Interpolating human bbox...")
    linked_bboxes = process_detections(detections, bbox_dist_threshold)
    linked_bboxes = interpolate_missing_bboxes(linked_bboxes)

    for frame_idx in tqdm(range(video_stream.frame_count)):
        ret, frame = read_frame(video_stream.cap, frame_idx)
        if not ret:
            break

        mask_bbox = get_masked_frame(
            frame,
            linked_bboxes.get(frame_idx, []),
            video_stream.frame_width,
            video_stream.frame_height,
        )
        video_stream.frames[frame_idx].idx = frame_idx  # Assign frame index
        video_stream.frames[frame_idx].frame = frame  # Store original frame
        video_stream.frames[frame_idx].bbox = linked_bboxes.get(frame_idx, [])
        video_stream.frames[frame_idx].bbox_mask = mask_bbox

    return video_stream
