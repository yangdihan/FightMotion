# pose.py
import numpy as np
import cv2

from constants import POSE_CONF_THRESHOLD

class Pose:
    def __init__(self, keypoints, track_id, frame, bbox):
        self.keypoints = keypoints
        self.track_id = track_id
        self.frame = frame
        self.bbox = bbox  # Add bbox to the constructor
        self.seq_length = keypoints.shape[0]  # Number of keypoints sequences
        self.pct_skin = self.calculate_pct_skin()

    def calculate_pct_skin(self):
        torso_polygon = self.get_torso_polygon()
        if torso_polygon is None:
            return 0.0

        mask = np.zeros(self.frame.pixels.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [torso_polygon], 1)
        
        # Apply mask to the frame
        torso_pixels = cv2.bitwise_and(self.frame.pixels, self.frame.pixels, mask=mask)
        
        # Detect skin within the masked torso area
        skin_mask = self.detect_skin(torso_pixels)
        skin_pixel_count = np.sum(skin_mask > 0)
        
        # Total torso pixels is the sum of the mask
        total_torso_pixels = np.sum(mask)
        if total_torso_pixels == 0:
            return 0.0
        
        pct_skin = (skin_pixel_count / total_torso_pixels) * 100  # Convert to percentage
        
        return pct_skin

    def detect_skin(self, img):
        # Convert image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Threshold the HSV image to get only skin colors
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        return skin_mask

    def get_torso_polygon(self):
        def get_fallback_keypoint(primary, *fallbacks, bbox_corner):
            for point in (primary, *fallbacks):
                if point[2] > POSE_CONF_THRESHOLD:
                    return (int(point[0]), int(point[1]))
            return bbox_corner

        keypoints = self.keypoints[-1].cpu().numpy()
        bbox_x, bbox_y, bbox_w, bbox_h = self.bbox

        left_shoulder = get_fallback_keypoint(
            keypoints[5], keypoints[7], keypoints[9], bbox_corner=(int(bbox_x), int(bbox_y))
        )
        right_shoulder = get_fallback_keypoint(
            keypoints[6], keypoints[8], keypoints[10], bbox_corner=(int(bbox_x + bbox_w), int(bbox_y))
        )
        left_hip = get_fallback_keypoint(
            keypoints[11], keypoints[13], keypoints[15], bbox_corner=(int(bbox_x), int(bbox_y + bbox_h))
        )
        right_hip = get_fallback_keypoint(
            keypoints[12], keypoints[14], keypoints[16], bbox_corner=(int(bbox_x + bbox_w), int(bbox_y + bbox_h))
        )

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        return np.array([left_shoulder, right_shoulder, right_hip, left_hip], np.int32)

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
