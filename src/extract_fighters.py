import numpy as np
from scipy.interpolate import interp1d
import torch
import torchvision
import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval().to(DEVICE)
YOLO_MODEL.classes = [0]  # Set model to detect only people (class 0)

# Load Mask R-CNN model
# mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().cuda()
def extract_person_yolo(frame, yolo_model, threshold, min_area):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(img)
    bboxes = []

    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if cls == 0 and conf >= threshold:
            x1, y1, x2, y2 = map(int, box)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area >= min_area:
                bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return bboxes

def bbox_dist(bbox1, bbox2):
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def diagonal_length(bbox):
        _, _, w, h = bbox
        return np.sqrt(w**2 + h**2)

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    vertices1 = [(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1), (x1 + w1 / 2, y1 + h1 / 2)]
    vertices2 = [(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2), (x2 + w2 / 2, y2 + h2 / 2)]

    avg_diagonal = (diagonal_length(bbox1) + diagonal_length(bbox2)) / 2
    avg_distance = sum(euclidean_distance(v1, v2) for v1, v2 in zip(vertices1, vertices2)) / 5

    return avg_distance / avg_diagonal

class LinkedBbox:
    def __init__(self, bbox=None, frame=None):
        self.bbox = bbox
        self.prev = None
        self.next = None
        self.frame = frame
        self.is_interpolated = False


def interpolate_bbox(start_bbox, end_bbox, steps):
    interpolated_bboxes = []
    for i in range(1, steps + 1):
        ratio = i / (steps + 1)
        interpolated_bbox = (
            start_bbox[0] * (1 - ratio) + end_bbox[0] * ratio,
            start_bbox[1] * (1 - ratio) + end_bbox[1] * ratio,
            start_bbox[2] * (1 - ratio) + end_bbox[2] * ratio,
            start_bbox[3] * (1 - ratio) + end_bbox[3] * ratio,
        )
        interpolated_bboxes.append(interpolated_bbox)
    return interpolated_bboxes

def make_connections(detections, frame_count, bbox_dist_threshold):
    linked_bboxes = {}

    # Initialize linked list nodes for each detected bounding box
    for frame_idx, bboxes in detections:
        linked_bboxes[frame_idx] = [LinkedBbox(bbox=bbox, frame=frame_idx) for bbox in bboxes]

    # Connect bounding boxes in consecutive frames
    for i in range(frame_count - 1):
        if i in linked_bboxes and i + 1 in linked_bboxes:
            current_bboxes = linked_bboxes[i]
            next_bboxes = linked_bboxes[i + 1]

            for cb in current_bboxes:
                min_dist = float('inf')
                best_match = None
                for nb in next_bboxes:
                    if nb.prev is None:
                        dist = bbox_dist(cb.bbox, nb.bbox)
                        if dist < min_dist and dist <= bbox_dist_threshold:
                            min_dist = dist
                            best_match = nb

                if best_match:
                    cb.next = best_match
                    best_match.prev = cb

    # Interpolate missing bounding boxes
    for frame_idx in range(frame_count):
        if frame_idx not in linked_bboxes:
            prev_frame = frame_idx - 1
            next_frame = frame_idx + 1

            while prev_frame >= 0 and prev_frame not in linked_bboxes:
                prev_frame -= 1

            while next_frame < frame_count and next_frame not in linked_bboxes:
                next_frame += 1

            if prev_frame >= 0 and next_frame < frame_count:
                prev_bboxes = [node for node in linked_bboxes[prev_frame]]
                next_bboxes = [node for node in linked_bboxes[next_frame]]

                for pb in prev_bboxes:
                    if pb.next is None:
                        min_dist = float('inf')
                        best_match = None
                        for nb in next_bboxes:
                            if nb.prev is None:
                                dist = bbox_dist(pb.bbox, nb.bbox)
                                if dist < min_dist and dist <= bbox_dist_threshold:
                                    min_dist = dist
                                    best_match = nb

                        if best_match:
                            steps = next_frame - prev_frame - 1
                            interpolated = interpolate_bbox(pb.bbox, best_match.bbox, steps)
                            for idx, bbox in enumerate(interpolated):
                                interpolated_node = LinkedBbox(bbox=bbox, frame=prev_frame + idx + 1)
                                interpolated_node.is_interpolated = True
                                if prev_frame + idx + 1 not in linked_bboxes:
                                    linked_bboxes[prev_frame + idx + 1] = []
                                linked_bboxes[prev_frame + idx + 1].append(interpolated_node)
                                pb.next = interpolated_node
                                interpolated_node.prev = pb
                                pb = interpolated_node
                            pb.next = best_match
                            best_match.prev = pb

    return linked_bboxes

def main(input_video_path, output_video_path, yolo_threshold, min_area_ratio, bbox_dist_threshold):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    min_area = min_area_ratio * frame_width * frame_height

    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        bboxes = extract_person_yolo(frame, YOLO_MODEL, yolo_threshold, min_area)
        detections.append((frame_idx, bboxes))

    cap.release()

    linked_bboxes = make_connections(detections, frame_count, bbox_dist_threshold)

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_idx in linked_bboxes:
            for linked_bbox in linked_bboxes[frame_idx]:
                x, y, w, h = map(int, linked_bbox.bbox)
                color = (0, 255, 0) if not linked_bbox.is_interpolated else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = 'Tracked' if not linked_bbox.is_interpolated else 'Interpolated'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete, output saved to", output_video_path)


if __name__ == "__main__":
    input_video_path = 'D:/Documents/devs/fight_motion/data/raw/aldo_holloway_single_angle.mp4'
    output_video_path = 'D:/Documents/devs/fight_motion/data/interim/aldo_holloway_yolo_conn_test.mp4'
    main(input_video_path, output_video_path, yolo_threshold=0.3, min_area_ratio=0.05, bbox_dist_threshold=0.1)
