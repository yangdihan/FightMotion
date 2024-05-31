from tqdm import tqdm
import numpy as np
import torch
import cv2

from constants import DEVICE, MRCNN_MODEL
from bbox import Bbox


class Contour:
    def __init__(self, frame_idx=None, geometry=None, confidence=None):
        self.frame_idx = frame_idx
        self.geometry = geometry
        self.bbox_vertices = self.compute_bbox_vertices()
        self.bbox_centroid = self.compute_bbox_centroid()
        self.confidence = confidence
        self.skin_pct = None
        self.trunk_color = None
        self.score = None
        self.pose = None

    def compute_bbox_vertices(self):
        x, y, w, h = cv2.boundingRect(self.geometry)
        return np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])

    def compute_bbox_centroid(self):
        x, y, w, h = cv2.boundingRect(self.geometry)
        return np.array([x + w / 2, y + h / 2])

    @staticmethod
    def draw_contours_on_mask(contours, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        for contour in contours:
            cv2.drawContours(
                mask, [contour.geometry], -1, (255, 255, 255), thickness=cv2.FILLED
            )
        return mask

    @staticmethod
    def significant_drop(prev_count, curr_count, drop_ratio):
        return prev_count is not None and curr_count < prev_count * drop_ratio

    @staticmethod
    def detect_skin(frame, contour):
        # Create a mask for the contour shape
        mask = Contour.draw_contours_on_mask([contour], frame.shape[:2])
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour.geometry)

        # Only check the upper 60% of the contour
        bottom_y = y + int(h * 0.618)

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

    @staticmethod
    def detect_trunk_color(frame, contour):
        # Create a mask for the contour shape
        mask = Contour.draw_contours_on_mask([contour], frame.shape[:2])
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour.geometry)

        # Only check the lower 40% of the contour
        top_y = y + int(h * 0.618)

        # Mask out the top 60%
        mask[:top_y] = 0

        # Apply the mask to the frame
        cropped_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Calculate the mean color of the trunk area
        trunk_color = cv2.mean(cropped_frame, mask=mask)[:3]

        return trunk_color

    @staticmethod
    def evaluate_fighter_likelihood(frame, contour):
        # Calculate skin exposure in the upper 60% of the contour
        skin_percentage = Contour.detect_skin(frame, contour)

        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour.geometry)

        # Calculate bounding box area
        bbox_area = w * h

        # Normalize the bounding box area (you may need to adjust this normalization factor based on your video resolution)
        normalized_bbox_area = bbox_area / (frame.shape[0] * frame.shape[1])

        return (skin_percentage, normalized_bbox_area)

    @staticmethod
    def extract_person_rcnn(frame, rcnn_model, min_confidence):
        pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = (
            torch.tensor(pil_img)
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # Perform Mask R-CNN detection
        with torch.no_grad():
            results = rcnn_model(pil_img)

        contours_with_likelihood = []

        for idx in range(len(results[0]["masks"])):
            score = results[0]["scores"][idx].item()
            if score < min_confidence:
                continue

            mask_rcnn = results[0]["masks"][idx, 0].mul(255).byte().cpu().numpy()
            contours, _ = cv2.findContours(
                mask_rcnn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                contours_with_likelihood.append(
                    Contour(frame_idx=None, geometry=contour, confidence=score)
                )

        return contours_with_likelihood

    @staticmethod
    def extract_fighter_contour(frame, rcnn_model, min_confidence):
        contours = Contour.extract_person_rcnn(frame, rcnn_model, min_confidence)

        if not contours:
            return [], []

        contours_with_likelihood = [
            (contour, Contour.evaluate_fighter_likelihood(frame, contour))
            for contour in contours
        ]

        # Sort contours by combined likelihood and keep only the top two
        contours_with_likelihood.sort(
            key=lambda x: x[1][0] * 0.618 + x[1][1] * 0.382, reverse=True
        )
        top_contours = [c for c, _ in contours_with_likelihood][:2]

        return top_contours

    @staticmethod
    def run_rcnn_contour(
        video_stream, rcnn_threshold, significant_drop_ratio, bbox_dist_threshold
    ):
        print(f"Detecting fighter contour by RCNN...")

        previous_non_blank_pixel_count = None
        top_contours_last = []

        for frame_idx in tqdm(range(video_stream.frame_count)):
            ret, frame_data = video_stream.read_frame(frame_idx)
            if not ret:
                break

            top_contours = Contour.extract_fighter_contour(
                video_stream.frames[frame_idx].mask_bbox, MRCNN_MODEL, rcnn_threshold
            )

            mask_contour = Contour.draw_contours_on_mask(top_contours, frame_data.shape)
            non_blank_pixel_count = cv2.countNonZero(
                cv2.cvtColor(mask_contour, cv2.COLOR_BGR2GRAY)
            )

            if Contour.significant_drop(
                previous_non_blank_pixel_count,
                non_blank_pixel_count,
                significant_drop_ratio,
            ):
                top_contours = video_stream.infer_missing_contours(
                    top_contours_last, top_contours, bbox_dist_threshold
                )
                mask_contour = Contour.draw_contours_on_mask(
                    top_contours, frame_data.shape
                )

            video_stream.frames[frame_idx].contours = top_contours
            video_stream.frames[frame_idx].mask_contour = mask_contour
            video_stream.frames[frame_idx].frame = cv2.bitwise_and(
                frame_data, mask_contour
            )

            previous_non_blank_pixel_count = non_blank_pixel_count
            top_contours_last = top_contours

        return video_stream
