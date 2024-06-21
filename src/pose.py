import numpy as np
import torch
import cv2

from constants import POSE_CONF_THRESHOLD


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
        self.keypoints = keypoints
        self.track_id = track_id
        self.frame = frame
        self.bbox = bbox  # Add bbox to the constructor
        self.seq_length = keypoints.shape[0]  # Number of keypoints sequences
        self.torso_polygon = sort_vertices_clockwise(self.get_torso_polygon())  # Save torso polygon
        self.pants_polygon = sort_vertices_clockwise(self.get_pants_polygon())  # Save pants polygon
        self.pct_skin = self.calculate_pct_skin()
        self.pct_pants, self.pants_color = self.calculate_pct_pants()

    @staticmethod
    def get_fallback_keypoint(primary, *fallbacks, bbox_corner):
        for point in (primary, *fallbacks):
            if (point[0]>0 or point[1]>0) and point[2] > POSE_CONF_THRESHOLD:
                return (int(point[0]), int(point[1]))
        return bbox_corner

    @staticmethod
    def find_most_prevalent_color(hsv_img, mask, skin_mask):
        # Convert images to tensor
        hsv_img = torch.tensor(hsv_img, device="cuda").permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask, device="cuda").unsqueeze(0).float()
        skin_mask = torch.tensor(skin_mask, device="cuda").unsqueeze(0).float()

        # Apply mask and skin mask to hsv_img
        hsv_img_masked = hsv_img * mask * (skin_mask == 0)

        # Get bounding box of the masked region
        nonzero_coords = torch.nonzero(mask)
        min_coords = nonzero_coords.min(dim=0)[0]
        max_coords = nonzero_coords.max(dim=0)[0]
        min_y, min_x = min_coords[0], min_coords[1]
        max_y, max_x = max_coords[0], max_coords[1]

        # Crop the hsv_img and mask to the bounding box
        hsv_img_cropped = hsv_img_masked[:, min_y:max_y + 1, min_x:max_x + 1]

        # Calculate the new size to downsample
        _, height, width = hsv_img_cropped.shape
        max_dim = max(height, width)
        if max_dim > 8:
            scale_factor = 8 / max_dim
            new_height = max(1, int(height * scale_factor))
            new_width = max(1, int(width * scale_factor))
            hsv_img_resized = torch.nn.functional.interpolate(hsv_img_cropped.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
        else:
            hsv_img_resized = hsv_img_cropped

        # Define color ranges in HSV
        color_ranges = {
            "red": [
                (torch.tensor([0, 0.2, 0.2], device="cuda"), torch.tensor([0.04, 1, 1], device="cuda")),
                (torch.tensor([0.67, 0.2, 0.2], device="cuda"), torch.tensor([1, 1, 1], device="cuda"))
            ],
            "blue": [(torch.tensor([0.55, 0.2, 0.2], device="cuda"), torch.tensor([0.78, 1, 1], device="cuda"))],
            "green": [(torch.tensor([0.22, 0.2, 0.2], device="cuda"), torch.tensor([0.44, 1, 1], device="cuda"))],
            "yellow": [(torch.tensor([0.11, 0.2, 0.2], device="cuda"), torch.tensor([0.16, 1, 1], device="cuda"))],
            "black": [(torch.tensor([0, 0, 0], device="cuda"), torch.tensor([1, 1, 0.2], device="cuda"))],
            "white": [(torch.tensor([0, 0, 0.8], device="cuda"), torch.tensor([1, 0.2, 1], device="cuda"))]
        }

        max_count = 0
        most_prevalent_color = None

        # Calculate the most prevalent color
        for color, ranges in color_ranges.items():
            count = 0
            for lower, upper in ranges:
                color_mask = ((hsv_img_resized >= lower[:, None, None]) & (hsv_img_resized <= upper[:, None, None])).all(dim=0)
                color_count = torch.sum(color_mask)
                count += color_count.item()

            if count > max_count:
                max_count = count
                most_prevalent_color = color

        return most_prevalent_color

    def calculate_pct_skin(self):
        # torso_polygon = self.get_torso_polygon()
        if self.torso_polygon is None:
            return 0.0

        mask = np.zeros(self.frame.pixels.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.torso_polygon], 1)

        # Apply mask to the frame
        torso_pixels = cv2.bitwise_and(self.frame.pixels, self.frame.pixels, mask=mask)

        # Detect skin within the masked torso area
        skin_mask = self.detect_skin(torso_pixels)
        # print(np.max(skin_mask), np.min(skin_mask))
        skin_pixel_count = np.sum((skin_mask > 0) & (mask > 0))

        # Total torso pixels is the sum of the mask
        total_torso_pixels = np.sum(mask)
        if total_torso_pixels == 0:
            return 0.0

        pct_skin = (
            skin_pixel_count / total_torso_pixels
        ) * 100  # Convert to percentage

        return pct_skin

    def calculate_pct_pants(self):
        # pants_polygon = self.get_pants_polygon()
        if self.pants_polygon is None:
            return 0.0

        mask = np.zeros(self.frame.pixels.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.pants_polygon], 1)

        # Apply mask to the frame
        pants_pixels = cv2.bitwise_and(self.frame.pixels, self.frame.pixels, mask=mask)

        # Detect skin within the masked pants area
        skin_mask = self.detect_skin(pants_pixels)

        # Invert the skin mask to get the non-skin (pants) mask
        # non_skin_mask = cv2.bitwise_not(skin_mask)

        # Count non-skin (pants) pixels only within the masked area
        pants_pixel_count = np.sum((skin_mask == 0) & (mask > 0))

        # Total pants pixels is the sum of the mask
        total_pants_pixels = np.sum(mask)
        if total_pants_pixels == 0:
            return 0.0

        # print(self.frame.idx, pants_pixel_count, total_pants_pixels)
        pct_pants = (
            pants_pixel_count / total_pants_pixels
        ) * 100  # Convert to percentage

        # Find the most prevalent color in the pants region
        hsv_pants = cv2.cvtColor(pants_pixels, cv2.COLOR_BGR2HSV)
        most_prevalent_color = Pose.find_most_prevalent_color(
            hsv_pants, mask, skin_mask
        )
        # most_prevalent_color = 'nope'

        return pct_pants, most_prevalent_color

    def detect_skin(self, img):
        # Convert image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)

        # Threshold the HSV image to get only skin colors
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

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

    def get_pants_polygon(self):
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
        
        # if self.frame.idx==71:
        #     if self.track_id==3:
        #         print(self.bbox)
        #         print([left_hip, right_hip, left_knee, right_knee])

        return np.array([left_hip, right_hip, right_knee, left_knee], np.int32)

    def plot_skeleton_kpts(self, im):

        palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ]
        )

        skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

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
