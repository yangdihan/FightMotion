# pose.py
import numpy as np
import cv2


class Pose:
    def __init__(self, keypoints, track_id, frame_idx):
        self.keypoints = keypoints
        self.track_id = track_id
        self.frame_idx = frame_idx
        self.seq_length = keypoints.shape[0]  # Number of keypoints sequences

    def plot_skeleton_kpts(self, im, steps):
        kptThres = 0.1
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

        # Print shape for debugging
        # print(f"Keypoints shape: {self.keypoints.shape}")

        if self.keypoints.ndim == 3:
            keypoints_flat = self.keypoints[-1, :, :].flatten()
        elif self.keypoints.ndim == 2:
            keypoints_flat = self.keypoints.flatten()
        else:
            print("Unexpected keypoints shape:", self.keypoints.shape)
            return im

        num_kpts = len(keypoints_flat) // steps

        for kid in range(num_kpts):
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord = (
                keypoints_flat[steps * kid],
                keypoints_flat[steps * kid + 1],
            )
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = keypoints_flat[steps * kid + 2]
                    if conf < kptThres:
                        continue
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
                int(keypoints_flat[(sk[0] - 1) * steps]),
                int(keypoints_flat[(sk[0] - 1) * steps + 1]),
            )
            pos2 = (
                int(keypoints_flat[(sk[1] - 1) * steps]),
                int(keypoints_flat[(sk[1] - 1) * steps + 1]),
            )
            if steps == 3:
                conf1 = keypoints_flat[(sk[0] - 1) * steps + 2]
                conf2 = keypoints_flat[(sk[1] - 1) * steps + 2]
                if conf1 < kptThres or conf2 < kptThres:
                    continue
            if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        return im
