import hashlib
import numpy as np


class Bbox:
    def __init__(self, xywh, frame, is_interpolated=False):
        self.xywh = xywh
        x, y, w, h = self.xywh
        self.area = w * h
        self.vertices = np.array(
            [
                [x - w / 2, y - h / 2],
                [x + w / 2, y - h / 2],
                [x - w / 2, y + h / 2],
                [x + w / 2, y + h / 2],
            ]
        )
        # self.centroid = np.array([x + w / 2, y + h / 2])

        self.prev = None
        self.next = None
        self.frame = frame
        self.is_interpolated = is_interpolated

        # self.confidence = confidence
        self.hash = self.compute_hash()

        self.pose_yolo8 = None

    def compute_hash(self):
        bbox_str = f"{self.xywh}-{self.frame.idx}"
        return hashlib.md5(bbox_str.encode()).hexdigest()

    def copy(self, is_interpolated):
        copied_bbox = Bbox(self.xywh, self.frame, is_interpolated)
        copied_bbox.pose_yolo8 = self.pose_yolo8
        return copied_bbox

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

    @staticmethod
    def linear_interpolate_bbox(prev_bbox, next_bbox, frame):
        """
        Linearly interpolate the center coordinates (x, y) and dimensions (w, h) of bounding boxes
        between prev_bbox and next_bbox for a given frame.
        """
        start_idx = prev_bbox.frame.idx
        end_idx = next_bbox.frame.idx
        alpha = (frame.idx - start_idx) / (end_idx - start_idx)

        # Linear interpolation of x, y, w, h
        x = prev_bbox.xywh[0] * (1 - alpha) + next_bbox.xywh[0] * alpha
        y = prev_bbox.xywh[1] * (1 - alpha) + next_bbox.xywh[1] * alpha
        w = prev_bbox.xywh[2] * (1 - alpha) + next_bbox.xywh[2] * alpha
        h = prev_bbox.xywh[3] * (1 - alpha) + next_bbox.xywh[3] * alpha

        # Create a new interpolated bounding box
        new_bbox = Bbox([x, y, w, h], frame, is_interpolated=True)
        new_bbox.pose_yolo8 = (
            prev_bbox.pose_yolo8
        )  # Copy pose information to maintain consistency

        return new_bbox

    @staticmethod
    def hull_bbox(bbox1, bbox2, frame):
        x1, y1, w1, h1 = bbox1.xywh
        x2, y2, w2, h2 = bbox2.xywh

        x1_min = min(x1 - w1 / 2, x2 - w2 / 2)
        y1_min = min(y1 - h1 / 2, y2 - h2 / 2)
        x2_max = max(x1 + w1 / 2, x2 + w2 / 2)
        y2_max = max(y1 + h1 / 2, y2 + h2 / 2)

        center_x = (x1_min + x2_max) / 2
        center_y = (y1_min + y2_max) / 2
        width = x2_max - x1_min
        height = y2_max - y1_min

        new_xywh = [center_x, center_y, width, height]
        new_bbox = Bbox(new_xywh, frame, is_interpolated=True)
        new_bbox.pose_yolo8 = (
            bbox1.pose_yolo8
        )  # Assign the same pose to maintain consistency

        return new_bbox

    # @staticmethod
    # def bbox_dist(bbox1, bbox2):
    #     def euclidean_distance(p1, p2):
    #         return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    #     def diagonal_length(bbox):
    #         _, _, w, h = bbox.xywh
    #         return np.sqrt(w**2 + h**2)

    #     x1, y1, w1, h1 = bbox1.xywh
    #     x2, y2, w2, h2 = bbox2.xywh

    #     vertices1 = [
    #         (x1, y1),
    #         (x1 + w1, y1),
    #         (x1, y1 + h1),
    #         (x1 + w1, y1 + h1),
    #     ]
    #     vertices2 = [
    #         (x2, y2),
    #         (x2 + w2, y2),
    #         (x2, y2 + h2),
    #         (x2 + w2, y2 + h2),
    #     ]
    #     centroid1 = (x1 + w1 / 2, y1 + h1 / 2)
    #     centroid2 = (x2 + w2 / 2, y2 + h2 / 2)

    #     avg_diagonal = (diagonal_length(bbox1) + diagonal_length(bbox2)) / 2
    #     avg_distance = (
    #         (
    #             sum(euclidean_distance(v1, v2) for v1, v2 in zip(vertices1, vertices2))
    #             / 4
    #         )
    #         + (euclidean_distance(centroid1, centroid2))
    #     ) / 2

    #     return avg_distance / avg_diagonal
