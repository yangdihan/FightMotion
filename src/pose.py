import os
import numpy as np
import torch
import cv2

from constants import POSE_CONF_THRESHOLD, DEVICE, KEYPOINT_VISUAL, COLOR_RANGES


def sort_vertices_clockwise(vertices):
    # Calculate the centroid of the polygon
    centroid = np.mean(vertices, axis=0)

    # Compute the angle of each vertex relative to the centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])

    # Sort the vertices based on these angles
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]


def color_classifier(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_count = 0
    most_prevalent_color = None

    # Mark masked out pixels as -1
    hsv_img[hsv_img == [0, 0, 0]] = -1

    for color, ranges in COLOR_RANGES.items():
        count = 0
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv_img, lower, upper)
            color_count = cv2.countNonZero(color_mask & (hsv_img[..., 2] != -1))
            count += color_count

        if count > max_count:
            max_count = count
            most_prevalent_color = color

    return most_prevalent_color


class Pose:
    def __init__(self, keypoints, track_id, frame, bbox):
        self.keypoints = keypoints
        self.track_id = track_id
        self.frame = frame
        self.bbox = bbox  # Add bbox to the constructor
        self.seq_length = keypoints.shape[0]  # Number of keypoints sequences
        self.torso_polygon = sort_vertices_clockwise(self.get_torso_polygon())
        self.trunk_polygon = sort_vertices_clockwise(self.get_trunk_polygon())
        self.pct_skin = self.compute_pct_skin()
        self.trunk_color = None

    @staticmethod
    def get_fallback_keypoint(primary, *fallbacks, bbox_corner):
        for point in (primary, *fallbacks):
            if (point[0] > 0 or point[1] > 0) and point[2] > POSE_CONF_THRESHOLD:
                return (int(point[0]), int(point[1]))
        return bbox_corner

    # @staticmethod
    def find_most_prevalent_color(self, hsv_img, mask, skin_mask):
        # Convert images to tensor
        hsv_img = torch.tensor(hsv_img, device=DEVICE).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask, device=DEVICE).unsqueeze(0).float()
        skin_mask = torch.tensor(skin_mask, device=DEVICE).unsqueeze(0).float()

        # Apply mask to hsv_img
        # hsv_img_masked = hsv_img * mask
        hsv_img_masked = hsv_img * mask * (skin_mask == 0)

        # Convert hsv_img_masked back to numpy without marking pixels as -1
        hsv_img_masked_np = (
            hsv_img_masked.permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)

        # Use color_classifier to determine the most prevalent color
        most_prevalent_color = color_classifier(hsv_img_masked_np)

        # Export the masked image before classification
        # export_path = os.path.join(
        #     "D:/Documents/devs/fight_motion/data/interim/",
        #     f"trunk_{self.frame.idx}_{self.track_id}_{most_prevalent_color}.jpg",
        # )
        # hsv_img_bgr = cv2.cvtColor(hsv_img_masked_np, cv2.COLOR_HSV2BGR)
        # cv2.imwrite(export_path, hsv_img_bgr)

        return most_prevalent_color

    def compute_pct_skin(self):

        mask = np.zeros(self.frame.pixels.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.torso_polygon], 1)

        # Apply mask to the frame
        torso_pixels = cv2.bitwise_and(self.frame.pixels, self.frame.pixels, mask=mask)

        # Detect skin within the masked torso area
        skin_mask = self.detect_skin(torso_pixels)

        skin_pixel_count = np.sum((skin_mask > 0) & (mask > 0))

        # Total torso pixels is the sum of the mask
        total_torso_pixels = np.sum(mask)

        pct_skin = skin_pixel_count / total_torso_pixels

        return pct_skin

    def classify_trunk_color(self):

        mask = np.zeros(self.frame.pixels.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.trunk_polygon], 1)

        # Apply mask to the frame
        trunk_pixels = cv2.bitwise_and(self.frame.pixels, self.frame.pixels, mask=mask)

        # Detect skin within the masked trunk area
        skin_mask = self.detect_skin(trunk_pixels)

        # Get bounding box of the trunk polygon
        x, y, w, h = cv2.boundingRect(self.trunk_polygon)

        # Crop the trunk_pixels, mask, and skin_mask to the bounding box
        cropped_trunk_pixels = trunk_pixels[y : y + h, x : x + w]
        cropped_mask = mask[y : y + h, x : x + w]
        cropped_skin_mask = skin_mask[y : y + h, x : x + w]

        # Find the most prevalent color in the cropped trunk region
        hsv_trunk_cropped = cv2.cvtColor(cropped_trunk_pixels, cv2.COLOR_BGR2HSV)
        most_prevalent_color = self.find_most_prevalent_color(
            hsv_trunk_cropped, cropped_mask, cropped_skin_mask
        )

        self.trunk_color = most_prevalent_color
        return

    def detect_skin(self, img):
        # Convert image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only skin colors
        skin_mask = cv2.inRange(
            hsv, COLOR_RANGES["skin"][0][0], COLOR_RANGES["skin"][0][1]
        )

        return skin_mask

    def get_torso_polygon(self):
        keypoints = self.keypoints[-1].cpu().numpy()
        bbox_x, bbox_y, bbox_w, bbox_h = self.bbox

        left_shoulder = Pose.get_fallback_keypoint(
            keypoints[5],
            keypoints[7],
            keypoints[9],
            bbox_corner=(int(bbox_x), int(bbox_y)),
        )
        right_shoulder = Pose.get_fallback_keypoint(
            keypoints[6],
            keypoints[8],
            keypoints[10],
            bbox_corner=(int(bbox_x + bbox_w), int(bbox_y)),
        )
        left_hip = Pose.get_fallback_keypoint(
            keypoints[11],
            keypoints[13],
            keypoints[15],
            bbox_corner=(int(bbox_x), int(bbox_y + bbox_h)),
        )
        right_hip = Pose.get_fallback_keypoint(
            keypoints[12],
            keypoints[14],
            keypoints[16],
            bbox_corner=(int(bbox_x + bbox_w), int(bbox_y + bbox_h)),
        )

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        return np.array([left_shoulder, right_shoulder, right_hip, left_hip], np.int32)

    def get_trunk_polygon(self):
        keypoints = self.keypoints[-1].cpu().numpy()
        bbox_x, bbox_y, bbox_w, bbox_h = self.bbox

        left_hip = Pose.get_fallback_keypoint(
            keypoints[11], bbox_corner=(int(bbox_x), int(bbox_y + bbox_h / 2))
        )
        right_hip = Pose.get_fallback_keypoint(
            keypoints[12], bbox_corner=(int(bbox_x + bbox_w), int(bbox_y + bbox_h / 2))
        )
        left_knee = Pose.get_fallback_keypoint(
            keypoints[13],
            keypoints[15],
            bbox_corner=(int(bbox_x), int(bbox_y + bbox_h)),
        )
        right_knee = Pose.get_fallback_keypoint(
            keypoints[14],
            keypoints[16],
            bbox_corner=(int(bbox_x + bbox_w), int(bbox_y + bbox_h)),
        )

        if not all([left_hip, right_hip, left_knee, right_knee]):
            return None

        return np.array([left_hip, right_hip, right_knee, left_knee], np.int32)

    def plot_skeleton_kpts(self, im):

        palette = np.array(KEYPOINT_VISUAL["palette"])

        skeleton = KEYPOINT_VISUAL["skeleton"]

        pose_limb_color = palette[
            [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
        ]
        pose_kpt_color = palette[
            [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
        ]
        radius = 5

        if self.keypoints.ndim == 3:
            keypoints_flat = self.keypoints[-1, :, :].flatten()
        elif self.keypoints.ndim == 2:
            keypoints_flat = self.keypoints.flatten()
        else:
            print("Unexpected keypoints shape:", self.keypoints.shape)
            return im

        num_kpts = len(keypoints_flat) // 3

        for kid in range(num_kpts):
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord, conf = (
                keypoints_flat[3 * kid],
                keypoints_flat[3 * kid + 1],
                keypoints_flat[3 * kid + 2],
            )

            if ~(x_coord <= 0 and y_coord <= 0) and conf > POSE_CONF_THRESHOLD:
                cv2.circle(
                    im,
                    (int(x_coord), int(y_coord)),
                    radius,
                    (int(r), int(g), int(b)),
                    -1,
                )

        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (
                int(keypoints_flat[(sk[0] - 1) * 3]),
                int(keypoints_flat[(sk[0] - 1) * 3 + 1]),
            )
            conf1 = keypoints_flat[(sk[0] - 1) * 3 + 2]
            pos2 = (
                int(keypoints_flat[(sk[1] - 1) * 3]),
                int(keypoints_flat[(sk[1] - 1) * 3 + 1]),
            )
            conf2 = keypoints_flat[(sk[1] - 1) * 3 + 2]

            if ((pos1[0] > 0 or pos1[1] > 0) and conf1 > POSE_CONF_THRESHOLD) and (
                (pos2[0] > 0 or pos2[1] > 0) and conf2 > POSE_CONF_THRESHOLD
            ):

                cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        return im
