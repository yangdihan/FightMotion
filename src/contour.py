from tqdm import tqdm
import numpy as np
import torch
import cv2

from bbox import Bbox


class Contour(Bbox):
    def __init__(self, geometry, confidence, frame, is_interpolated=False):
        self.geometry = geometry

        super().__init__(
            cv2.boundingRect(self.geometry), confidence, frame, is_interpolated
        )

        self.skin_pct = None
        self.trunk_color = None
        self.score = None
        self.pose = None

        self.mask = self.frame.mask_frame_with_contours([self])
        self.mask_upper = self.mask.copy()
        self.mask_lower = self.mask.copy()

        x, y, w, h = self.xywh
        # Mask out the lower_body
        self.mask_upper[int(y + h * 0.618) :] = 0
        # Mask out the upper body
        self.mask_lower[: int(y + (1 - 0.618) * h)] = 0

        self.pixel_upper = self.frame.crop_frame_with_mask(self.mask_upper)
        self.pixel_lower = self.frame.crop_frame_with_mask(self.mask_lower)

    @property
    def bbox_upper(self):
        x, y, w, h = self.xywh
        return x, y, w, int(h * 0.618)

    @property
    def bbox_lower(self):
        x, y, w, h = self.xywh
        return x, int(y + (1 - 0.618) * h), w, int(h * 0.618)

    @staticmethod
    def compute_geometry_from_xywh(xywh):
        x, y, w, h = map(int, xywh)
        geometry = np.array(
            [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32
        )
        return geometry

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
        self.pct_skin = skin_area / total_area if total_area > 0 else 0

        return

    def detect_trunk_color(self):
        # Convert to HSV color space
        hsv = cv2.cvtColor(self.pixel_lower, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Remove skin color pixels
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        non_skin_mask = cv2.bitwise_not(skin_mask)
        non_skin_pixels = cv2.bitwise_and(
            self.pixel_lower, self.pixel_lower, mask=non_skin_mask
        )

        # Downsample the image to speed up processing
        downsample_factor = 8
        trunk_pixels_small = cv2.resize(
            non_skin_pixels,
            (
                non_skin_pixels.shape[1] // downsample_factor,
                non_skin_pixels.shape[0] // downsample_factor,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        non_skin_mask_small = cv2.resize(
            non_skin_mask,
            (
                non_skin_mask.shape[1] // downsample_factor,
                non_skin_mask.shape[0] // downsample_factor,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert to RGB for color detection
        trunk_pixels_rgb = cv2.cvtColor(trunk_pixels_small, cv2.COLOR_BGR2RGB)
        trunk_pixels_rgb = trunk_pixels_rgb[non_skin_mask_small > 0].reshape((-1, 3))

        # Define color boundaries in RGB
        color_boundaries = {
            "Red": ([200, 0, 0], [255, 50, 50]),
            "Green": ([0, 200, 0], [50, 255, 50]),
            "Blue": ([0, 0, 200], [50, 50, 255]),
            "Yellow": ([200, 200, 0], [255, 255, 50]),
            "White": ([200, 200, 200], [255, 255, 255]),
            "Black": ([0, 0, 0], [50, 50, 50]),
        }

        # Count the number of pixels within each color boundary
        color_counts = {color: 0 for color in color_boundaries.keys()}
        color_masks = {
            color: np.zeros_like(non_skin_mask) for color in color_boundaries.keys()
        }

        for pixel in tqdm(trunk_pixels_rgb):
            for color, (lower, upper) in color_boundaries.items():
                if all(lower[j] <= pixel[j] <= upper[j] for j in range(3)):
                    color_counts[color] += 1

        # Determine the most dominant color
        dominant_color = max(color_counts, key=color_counts.get)
        self.trunk_color = dominant_color

        # Generate trunk color mask
        trunk_mask_full_size = np.zeros_like(non_skin_mask)
        for color, (lower, upper) in color_boundaries.items():
            if color == dominant_color:
                trunk_mask_full_size = cv2.inRange(
                    cv2.cvtColor(self.pixel_lower, cv2.COLOR_BGR2RGB),
                    np.array(lower),
                    np.array(upper),
                )
                break

        self.trunk_color_mask = trunk_mask_full_size

        return
