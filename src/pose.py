# pose.py
import numpy as np
import cv2

from constants import POSE_CONF_THRESHOLD


class Pose:
    def __init__(self, keypoints, track_id, frame):
        self.keypoints = keypoints
        self.track_id = track_id
        self.frame = frame
        self.seq_length = keypoints.shape[0]  # Number of keypoints sequences

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
