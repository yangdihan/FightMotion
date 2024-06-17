import hashlib
import numpy as np


class Bbox:
    def __init__(self, xywh, frame, confidence, is_interpolated=False):
        self.xywh = xywh
        self.prev = None
        self.next = None
        self.frame = frame
        self.is_interpolated = is_interpolated
        self.vertices = self.compute_vertices()
        self.centroid = self.compute_centroid()
        self.confidence = confidence
        self.hash = self.compute_hash()

    def compute_hash(self):
        bbox_str = f"{self.xywh}-{self.frame.idx}"
        return hashlib.md5(bbox_str.encode()).hexdigest()

    def compute_vertices(self):
        x, y, w, h = self.xywh
        return np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])

    def compute_centroid(self):
        x, y, w, h = self.xywh
        return np.array([x + w / 2, y + h / 2])

    @staticmethod
    def bbox_dist(bbox1, bbox2):
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def diagonal_length(bbox):
            _, _, w, h = bbox.xywh
            return np.sqrt(w**2 + h**2)

        x1, y1, w1, h1 = bbox1.xywh
        x2, y2, w2, h2 = bbox2.xywh

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
            (
                sum(euclidean_distance(v1, v2) for v1, v2 in zip(vertices1, vertices2))
                / 4
            )
            + (euclidean_distance(centroid1, centroid2))
        ) / 2

        return avg_distance / avg_diagonal

    def expand_bbox(self, expand_ratio):
        x, y, w, h = self.xywh
        x -= expand_ratio * w
        y -= expand_ratio * h
        w = (1 + 2 * expand_ratio) * w
        h = (1 + 2 * expand_ratio) * h
        x = int(max(0, x))
        y = int(max(0, y))
        w = int(min(self.frame.pixels.shape[1] - x, w))
        h = int(min(self.frame.pixels.shape[0] - y, h))

        return x, y, w, h
