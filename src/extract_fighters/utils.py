import hashlib
import numpy as np
import cv2


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


def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
    ret, frame = cap.read()
    return ret, frame
