from tqdm import tqdm
import numpy as np
import torch
import cv2

from bbox import Bbox


class Contour(Bbox):
    def __init__(self, frame, geometry, confidence):
        self.geometry = geometry

        super().__init__(cv2.boundingRect(self.geometry), frame, confidence)

        self.skin_pct = None
        self.trunk_color = None
        self.score = None
        self.pose = None

        self.mask = self.frame.mask_frame_with_contours([self])
        self.mask_upper = self.mask.copy()
        self.mask_lower = self.mask.copy()

        x, y, w, h = self.xywh
        # Mask out the lower_body
        self.mask_upper[y + int(h * 0.618) :] = 0
        # Mask out the upper body
        self.mask_lower[: y + int(h * 0.618)] = 0

        self.pixel_upper = self.frame.crop_frame_with_mask(self.mask_upper)
        self.pixel_lower = self.frame.crop_frame_with_mask(self.mask_lower)

    @staticmethod
    def significant_drop(prev_count, curr_count, drop_ratio):
        return prev_count is not None and curr_count < prev_count * drop_ratio

    @staticmethod
    def infer_missing_contours(contours_last, contours_this, bbox_dist_threshold):

        paired_last = [False] * len(contours_last)
        paired_this = [False] * len(contours_this)

        for i, bbox_this in enumerate(contours_this):
            for j, bbox_last in enumerate(contours_last):
                if Bbox.bbox_dist(bbox_this, bbox_last) <= bbox_dist_threshold:
                    paired_this[i] = True
                    paired_last[j] = True

        # Add unpaired contours from the last frame to the current frame
        for i, paired in enumerate(paired_last):
            if not paired:
                contours_this.append(contours_last[i])

        return contours_this

    def estimate_skin_exposure(self):
        # Convert to HSV color space
        hsv = cv2.cvtColor(self.pixel_upper, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create a mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Calculate the percentage of skin area
        skin_area = cv2.countNonZero(skin_mask)
        total_area = cv2.countNonZero(self.mask)
        skin_percentage = skin_area / total_area if total_area > 0 else 0

        return skin_percentage

    # @staticmethod
    def detect_trunk_color(self):

        trunk_color = cv2.mean(self.pixel_lower, mask=self.lower_skin)[:3]

        return trunk_color

    # @staticmethod
    def evaluate_fighter_likelihood(self):
        # Calculate skin exposure in the upper 60% of the contour
        self.pct_skin = self.estimate_skin_exposure()

        x, y, w, h = self.xywh
        bbox_area = w * h

        # Normalize the bounding box area (you may need to adjust this normalization factor based on your video resolution)
        self.pct_bbox = bbox_area / (
            self.frame.pixels.shape[0] * self.frame.pixels.shape[1]
        )

        return
