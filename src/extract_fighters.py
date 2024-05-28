import os
import json
import hashlib
import numpy as np
from scipy.interpolate import interp1d
import torch
import torchvision
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model
YOLO_MODEL = (
    torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval().to(DEVICE)
)
YOLO_MODEL.classes = [0]  # Set model to detect only people (class 0)


# Load Mask R-CNN model
MRCNN_MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(DEVICE)


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

def extract_person_rcnn(frame, rcnn_model, min_confidence):
    pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = torch.tensor(pil_img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)

    # Perform Mask R-CNN detection
    with torch.no_grad():
        results = rcnn_model(pil_img)

    for idx in range(len(results[0]['masks'])):
        score = results[0]['scores'][idx].item()
        if score < min_confidence:
            continue

        mask_rcnn = results[0]['masks'][idx, 0].mul(255).byte().cpu().numpy()
        contours, _ = cv2.findContours(mask_rcnn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Label the confidence score
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(frame, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def bbox_dist(bbox1, bbox2):
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def diagonal_length(bbox):
        _, _, w, h = bbox
        return np.sqrt(w**2 + h**2)

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    vertices1 = [
        (x1, y1),
        (x1 + w1, y1),
        (x1, y1 + h1),
        (x1 + w1, y1 + h1),
    ]
    vertices2 = [
        (x2, y2),
        (x2 + w2, y2),
        (x2, y2 + h2),
        (x2 + w2, y2 + h2),
    ]
    centroid1 = (x1 + w1 / 2, y1 + h1 / 2)
    centroid2 = (x2 + w2 / 2, y2 + h2 / 2)

    avg_diagonal = (diagonal_length(bbox1) + diagonal_length(bbox2)) / 2
    avg_distance = (
        (sum(euclidean_distance(v1, v2) for v1, v2 in zip(vertices1, vertices2)) / 4)
        + (euclidean_distance(centroid1, centroid2))
    ) / 2

    return avg_distance / avg_diagonal


class LinkedBbox:
    def __init__(self, bbox=None, frame=None, is_interpolated=False):
        self.bbox = bbox
        self.prev = None
        self.next = None
        self.frame = frame
        self.is_interpolated = is_interpolated
        self.hash = self.compute_hash()

    def compute_hash(self):
        bbox_str = f"{self.bbox}-{self.frame}"
        return hashlib.md5(bbox_str.encode()).hexdigest()

    def to_dict(self):
        return {
            "bbox": self.bbox,
            "frame": self.frame,
            "is_interpolated": self.is_interpolated,
            "hash": self.hash,
            "prev": self.prev.hash if self.prev else None,
            "next": self.next.hash if self.next else None,
        }


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

def detect_skin(frame, contour):
    # Create a mask for the contour shape
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Only check the upper 60% of the contour
    top_y = y
    bottom_y = y + int(h * 0.6)

    # Mask out the bottom 40%
    mask[bottom_y:] = 0

    # Apply the mask to the frame
    cropped_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to HSV color space
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Calculate the percentage of skin area
    skin_area = cv2.countNonZero(skin_mask)
    total_area = cv2.countNonZero(mask)
    skin_percentage = skin_area / total_area if total_area > 0 else 0

    return skin_percentage

def evaluate_fighter_likelihood(frame, contour):
    # Calculate skin exposure in the upper 60% of the contour
    skin_percentage = detect_skin(frame, contour)

    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate bounding box area
    bbox_area = w * h

    # Heuristic: combine skin exposure and bounding box area
    # You can adjust the weights based on experimentation
    skin_weight = 0.7
    area_weight = 0.3

    # Normalize the bounding box area (you may need to adjust this normalization factor based on your video resolution)
    normalized_bbox_area = bbox_area / (frame.shape[0] * frame.shape[1])

    fighter_likelihood = skin_weight * skin_percentage + area_weight * normalized_bbox_area

    return fighter_likelihood

def extract_person_rcnn(frame, rcnn_model, min_confidence):
    pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = torch.tensor(pil_img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)

    # Perform Mask R-CNN detection
    with torch.no_grad():
        results = rcnn_model(pil_img)

    contours_with_likelihood = []

    for idx in range(len(results[0]['masks'])):
        score = results[0]['scores'][idx].item()
        if score < min_confidence:
            continue

        mask_rcnn = results[0]['masks'][idx, 0].mul(255).byte().cpu().numpy()
        contours, _ = cv2.findContours(mask_rcnn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            fighter_likelihood = evaluate_fighter_likelihood(frame, contour)
            contours_with_likelihood.append((contour, fighter_likelihood))

    # Sort contours by likelihood and keep only the top two
    contours_with_likelihood.sort(key=lambda x: x[1], reverse=True)
    top_contours = contours_with_likelihood[:2]

    mask = np.zeros_like(frame)
    for contour, likelihood in top_contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(frame, f"{likelihood:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_frame = cv2.bitwise_and(frame, mask)

    return result_frame


def main(
    input_video_path, output_folder, yolo_threshold, min_area_ratio, bbox_dist_threshold, rcnn_threshold
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    min_area = min_area_ratio * frame_width * frame_height

    detections = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = (
            int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        )  # Convert to 0-based indexing
        bboxes = extract_person_yolo(frame, YOLO_MODEL, yolo_threshold, min_area)
        detections[frame_idx] = bboxes

    cap.release()

    linked_bboxes = direct_connection(detections, bbox_dist_threshold)
    linked_bboxes = infer_connection(linked_bboxes, bbox_dist_threshold)
    linked_bboxes = interpolate_missing_bboxes(linked_bboxes)

    # Prepare the output video writer
    output_video_path = os.path.join(output_folder, "output_video_rcnn.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    cap = cv2.VideoCapture(input_video_path)  # Reopen the video to read frames again
    for frame_idx in range(frame_count):
        cap.set(
            cv2.CAP_PROP_POS_FRAMES, frame_idx + 1
        )  # Convert back to 1-based indexing
        ret, frame = cap.read()
        if not ret:
            break

        mask = np.zeros_like(frame)

        if frame_idx in linked_bboxes:
            for node in linked_bboxes[frame_idx]:
                x, y, w, h = map(int, node.bbox)
                # Expand bbox size by 5%
                x -= int(0.05 * w)
                y -= int(0.05 * h)
                w = int(1.1 * w)
                h = int(1.1 * h)
                x = max(0, x)
                y = max(0, y)
                w = min(frame_width - x, w)
                h = min(frame_height - y, h)
                mask[y : y + h, x : x + w] = frame[y : y + h, x : x + w]

        mask = extract_person_rcnn(mask, MRCNN_MODEL, rcnn_threshold)
        out.write(mask)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    linked_bboxes_json = {
        frame: [node.to_dict() for node in nodes]
        for frame, nodes in linked_bboxes.items()
    }

    with open(os.path.join(output_folder, "linked_bboxes.json"), "w") as f:
        json.dump(linked_bboxes_json, f, indent=4)

    print("Processing complete, frames and video saved to", output_folder)


if __name__ == "__main__":
    input_video_path = (
        "D:/Documents/devs/fight_motion/data/raw/aldo_holloway_single_angle.mp4"
    )
    output_folder = "D:/Documents/devs/fight_motion/data/interim/"
    main(
        input_video_path,
        output_folder,
        yolo_threshold=0.382,
        min_area_ratio=0.05,
        bbox_dist_threshold=0.1,
        rcnn_threshold=0.9,
    )
