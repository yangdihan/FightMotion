import os
import numpy as np
import torch
import cv2

from constants import (
    POSE_CONF_THRESHOLD,
    DEVICE,
    KEYPOINT_VISUAL,
    COLOR_RANGES,
    SIZE_SQUARE_IMG,
)


def sort_vertices_clockwise(vertices):
    # Calculate the centroid of the polygon
    centroid = np.mean(vertices, axis=0)

    # Compute the angle of each vertex relative to the centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])

    # Sort the vertices based on these angles
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]


class Pose:
    def __init__(self, keypoints, track_id, frame, bbox):
        self.keypoints = keypoints.cpu()
        self.track_id = track_id
        self.frame = frame
        self.bbox = bbox.cpu()  # Add bbox to the constructor
        self.seq_length = keypoints.shape[0]  # Number of keypoints sequences

        self.torso_polygon = self.get_torso_polygon()
        self.trunk_polygon = self.get_trunk_polygon()
        self.torso_hsv = self.polygon_to_square_img(self.torso_polygon)
        self.trunk_hsv = self.polygon_to_square_img(self.trunk_polygon)

        self.compute_pct_skin()

        self.trunk_id = -1
        self.fighter_id = -1
        return

    @staticmethod
    def get_fallback_keypoint(primary, *fallbacks, bbox_corner):
        for point in (primary, *fallbacks):
            if (point[0] > 0 or point[1] > 0) and point[2] > POSE_CONF_THRESHOLD:
                return (int(point[0]), int(point[1]))
        return bbox_corner

    def polygon_to_square_img(self, polygon_xy):
        # Define the destination points for the perspective transform
        dst_points = np.array(
            [
                [0, 0],
                [SIZE_SQUARE_IMG - 1, 0],
                [SIZE_SQUARE_IMG - 1, SIZE_SQUARE_IMG - 1],
                [0, SIZE_SQUARE_IMG - 1],
            ],
            dtype=np.float32,
        )

        # Ensure the torso_polygon is sorted clockwise
        polygon = np.array(sort_vertices_clockwise(polygon_xy), dtype=np.float32)

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(polygon, dst_points)

        # Apply the perspective transformation
        transformed_img = cv2.warpPerspective(
            # self.frame.pixels, M, (SIZE_SQUARE_IMG, SIZE_SQUARE_IMG)
            self.frame,
            M,
            (SIZE_SQUARE_IMG, SIZE_SQUARE_IMG),
        )

        img_hsv = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2HSV)

        return img_hsv

        # self.torso_img = torch.tensor(hsv_trunk, device=DEVICE).permute(2, 0, 1).float() / 255.0
        # return

    def compute_pct_skin(self):
        # Detect skin within the transformed torso area
        skin_mask = cv2.inRange(
            self.torso_hsv, COLOR_RANGES["skin"][0][0], COLOR_RANGES["skin"][0][1]
        )
        # Calculate the number of skin pixels and total pixels
        skin_pixel_count = np.sum(skin_mask > 0)
        total_torso_pixels = SIZE_SQUARE_IMG * SIZE_SQUARE_IMG

        # Calculate the percentage of skin pixels
        self.pct_skin = skin_pixel_count / total_torso_pixels

        return

    def get_torso_polygon(self):
        keypoints = self.keypoints.numpy()
        x, y, w, h = self.bbox

        left_shoulder = Pose.get_fallback_keypoint(
            keypoints[5],
            keypoints[7],
            keypoints[9],
            bbox_corner=(int(x - w / 2), int(y - h / 4)),
        )
        right_shoulder = Pose.get_fallback_keypoint(
            keypoints[6],
            keypoints[8],
            keypoints[10],
            bbox_corner=(int(x + w / 2), int(y - h / 4)),
        )
        left_hip = Pose.get_fallback_keypoint(
            keypoints[11],
            keypoints[13],
            keypoints[15],
            bbox_corner=(int(x - w / 2), int(y)),
        )
        right_hip = Pose.get_fallback_keypoint(
            keypoints[12],
            keypoints[14],
            keypoints[16],
            bbox_corner=(int(x + w / 2), int(y)),
        )

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        return np.array([left_shoulder, right_shoulder, right_hip, left_hip], np.int32)

    def get_trunk_polygon(self):
        keypoints = self.keypoints.cpu().numpy()
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]

        bbox_x, bbox_y, bbox_w, bbox_h = self.bbox
        bbox_points = [
            [bbox_x - bbox_w / 4, bbox_y],
            [bbox_x + bbox_w / 4, bbox_y],
            [bbox_x - bbox_w / 4, bbox_y + bbox_h / 4],
            [bbox_x + bbox_w / 4, bbox_y + bbox_h / 4],
        ]

        def is_valid(point):
            return (point[0] > 0 or point[1] > 0) and point[2] > POSE_CONF_THRESHOLD

        valid_points = [
            is_valid(left_hip),
            is_valid(right_hip),
            is_valid(left_knee),
            is_valid(right_knee),
        ]
        valid_count = sum(valid_points)

        if valid_count == 4:
            polygon_vertices = np.array(
                [left_hip[:2], right_hip[:2], right_knee[:2], left_knee[:2]]
            )

        elif valid_count == 3:
            if not valid_points[2]:  # left_knee missing
                center_x = (left_hip[0] + right_hip[0]) / 2
                if right_knee[0] < center_x:
                    left_knee = [max(left_hip[0], right_hip[0]), right_knee[1], 1]
                else:
                    left_knee = [min(left_hip[0], right_hip[0]), right_knee[1], 1]
            elif not valid_points[3]:  # right_knee missing
                center_x = (left_hip[0] + right_hip[0]) / 2
                if left_knee[0] < center_x:
                    right_knee = [max(left_hip[0], right_hip[0]), left_knee[1], 1]
                else:
                    right_knee = [min(left_hip[0], right_hip[0]), left_knee[1], 1]
            elif not valid_points[0]:  # left_hip missing
                center_x = (left_knee[0] + right_knee[0]) / 2
                if right_hip[0] < center_x:
                    left_hip = [max(left_knee[0], right_knee[0]), right_hip[1], 1]
                else:
                    left_hip = [min(left_knee[0], right_knee[0]), right_hip[1], 1]
            elif not valid_points[1]:  # right_hip missing
                center_x = (left_knee[0] + right_knee[0]) / 2
                if left_hip[0] < center_x:
                    right_hip = [max(left_knee[0], right_knee[0]), left_hip[1], 1]
                else:
                    right_hip = [min(left_knee[0], right_knee[0]), left_hip[1], 1]

            polygon_vertices = np.array(
                [left_hip[:2], right_hip[:2], right_knee[:2], left_knee[:2]]
            )

        elif valid_count == 2:
            if valid_points[0] and valid_points[2]:  # left_hip and left_knee
                right_hip = [right_knee[0], left_hip[1], 1]
                right_knee = [right_knee[0], left_knee[1], 1]
            elif valid_points[1] and valid_points[3]:  # right_hip and right_knee
                left_hip = [left_knee[0], right_hip[1], 1]
                left_knee = [left_knee[0], right_knee[1], 1]
            elif valid_points[0] and valid_points[1]:  # both hips
                left_knee = [bbox_points[2][0], bbox_points[2][1], 1]
                right_knee = [bbox_points[3][0], bbox_points[3][1], 1]
            elif valid_points[2] and valid_points[3]:  # both knees
                left_hip = [bbox_points[0][0], bbox_points[0][1], 1]
                right_hip = [bbox_points[1][0], bbox_points[1][1], 1]

            polygon_vertices = np.array(
                [left_hip[:2], right_hip[:2], right_knee[:2], left_knee[:2]]
            )

        elif valid_count == 1:
            valid_point = [
                point
                for point, valid in zip(
                    [left_hip, right_hip, left_knee, right_knee], valid_points
                )
                if valid
            ][0]

            bbox_points = sorted(
                bbox_points,
                key=lambda p: np.linalg.norm(np.array(p) - np.array(valid_point[:2])),
            )
            polygon_vertices = np.array([valid_point[:2], *bbox_points[:3]])

        else:
            polygon_vertices = np.array(bbox_points, np.int32)

        # Calculate the midpoint for the left and right sides
        mid_left = (polygon_vertices[0] + polygon_vertices[3]) / 2
        mid_right = (polygon_vertices[1] + polygon_vertices[2]) / 2

        # Update the bottom-left and bottom-right vertices
        polygon_vertices[2] = mid_right
        polygon_vertices[3] = mid_left

        return sort_vertices_clockwise(polygon_vertices).astype(np.int32)

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


# def color_classifier(img):
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     max_count = 0
#     most_prevalent_color = None

#     # Mark masked out pixels as -1
#     hsv_img[hsv_img == [0, 0, 0]] = -1

#     for color, ranges in COLOR_RANGES.items():
#         count = 0
#         for lower, upper in ranges:
#             color_mask = cv2.inRange(hsv_img, lower, upper)
#             color_count = cv2.countNonZero(color_mask & (hsv_img[..., 2] != -1))
#             count += color_count

#         if count > max_count:
#             max_count = count
#             most_prevalent_color = color

#     return most_prevalent_color


# @staticmethod
# def find_most_prevalent_color(hsv_img, mask, skin_mask):
#     # Convert images to tensor
#     hsv_img = torch.tensor(hsv_img, device=DEVICE).permute(2, 0, 1).float() / 255.0
#     mask = torch.tensor(mask, device=DEVICE).unsqueeze(0).float()
#     skin_mask = torch.tensor(skin_mask, device=DEVICE).unsqueeze(0).float()

#     # Apply mask to hsv_img
#     # hsv_img_masked = hsv_img * mask
#     hsv_img_masked = hsv_img * mask * (skin_mask == 0)

#     # Convert hsv_img_masked back to numpy without marking pixels as -1
#     hsv_img_masked_np = (
#         hsv_img_masked.permute(1, 2, 0).cpu().numpy() * 255
#     ).astype(np.uint8)

#     # Use color_classifier to determine the most prevalent color
#     most_prevalent_color = color_classifier(hsv_img_masked_np)

#     # Export the masked image before classification
#     # export_path = os.path.join(
#     #     "D:/Documents/devs/fight_motion/data/interim/",
#     #     f"trunk_{self.frame.idx}_{self.track_id}_{most_prevalent_color}.jpg",
#     # )
#     # hsv_img_bgr = cv2.cvtColor(hsv_img_masked_np, cv2.COLOR_HSV2BGR)
#     # cv2.imwrite(export_path, hsv_img_bgr)

#     return most_prevalent_color
