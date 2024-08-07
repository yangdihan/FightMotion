import os
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from hmmlearn import hmm
import torch
import cv2
import pyopenpose as op

from constants import (
    POSE_TRACKER,
    # SIZE_SQUARE_IMG,
    MIN_APPEARING_FRAMES,
    OPENPOSE_PARAM,
)
from frame import Frame
from bbox import Bbox
from pose import Pose


DIR_IN = "D:/Documents/devs/fight_motion/data/raw/"
DIR_OUT = "D:/Documents/devs/fight_motion/data/interim/"


class Clip:
    def __init__(self, fn_video) -> None:

        self.clip_name = fn_video

        self.cap = cv2.VideoCapture(os.path.join(DIR_IN, f"{self.clip_name}.mp4"))
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = [None] * self.frame_count

        print(f"Parsing video clip `{fn_video}`...")
        for frame_idx in tqdm(range(self.frame_count)):
            ret, pixels = self.read_frame(frame_idx)
            if not ret:
                break
            self.frames[frame_idx] = Frame(frame_idx, pixels)

        self.cap.release()
        return

    def read_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return ret, frame

    def output(self, jpg=True):
        out = cv2.VideoWriter(
            os.path.join(DIR_OUT, f"{self.clip_name}_poses_{POSE_TRACKER}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        poses_json_data = []

        print("Exporting frames...")
        for frame in tqdm(self.frames):
            marked_frame = frame.pixels.copy()
            frame_poses = []

            for pose in frame.poses:
                marked_frame = pose.plot_skeleton_kpts(marked_frame)
                text = (
                    # f"track:{pose.track_id}, trunk:{pose.trunk_id}, {int(pose.pct_skin*100)}%, {pose.trunk_color}"
                    f"track:{pose.track_id}, trunk:{pose.trunk_id}, {int(pose.pct_skin*100)}%"
                )

                keypoints = (
                    pose.keypoints[-1].cpu().numpy()
                )  # Get the last set of keypoints
                x = int(np.median(keypoints[:, 0]))
                y = int(np.median(keypoints[:, 1]))

                cv2.putText(
                    marked_frame,
                    text,
                    (x, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            poses_json_data[frame.idx] = {
                int(trunk_id): keypoints for trunk_id, keypoints in frame_poses.items()
            }

            poses_json_data.append({"frame": frame.idx, "poses": frame_poses})
            out.write(marked_frame)

            if jpg:
                frame_output_path = os.path.join(
                    DIR_OUT, f"{self.clip_name}_frame{frame.idx}.jpg"
                )
                cv2.imwrite(frame_output_path, marked_frame)

        out.release()

        with open(
            os.path.join(DIR_OUT, f"{self.clip_name}_poses_{POSE_TRACKER}.json"), "w"
        ) as json_file:
            json.dump(poses_json_data, json_file, indent=4)

        return

    # def output_2bbox(self):
    #     out_fighter_0 = cv2.VideoWriter(
    #         os.path.join(DIR_OUT, f"{self.clip_name}_fighter_0.mp4"),
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         self.fps,
    #         (self.frame_width, self.frame_height),
    #     )

    #     out_fighter_1 = cv2.VideoWriter(
    #         os.path.join(DIR_OUT, f"{self.clip_name}_fighter_1.mp4"),
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         self.fps,
    #         (self.frame_width, self.frame_height),
    #     )

    #     # poses_json_data = {}

    #     print("Exporting frames...")
    #     for frame in tqdm(self.frames):
    #         # marked_frame = frame.pixels.copy()
    #         fighter_0_frame = np.zeros_like(frame.pixels)
    #         fighter_1_frame = np.zeros_like(frame.pixels)
    #         # frame_poses = {}

    #         for bbox in frame.bboxes:
    #             pose = bbox.pose_yolo8
    #             # marked_frame = pose.plot_skeleton_kpts(marked_frame)
    #             # keypoints = (
    #             #     pose.keypoints[-1].cpu().numpy()
    #             # )  # Get the last set of keypoints

    #             # text = f"frame:{frame.idx}, trunk:{pose.trunk_id}, hmm: {pose.hmm_id}, {int(pose.pct_skin*100)}%"
    #             # x = int(np.median(keypoints[:, 0]))
    #             # y = int(np.median(keypoints[:, 1]))
    #             # cv2.putText(
    #             #     marked_frame,
    #             #     text,
    #             #     (x, y + 15),
    #             #     cv2.FONT_HERSHEY_SIMPLEX,
    #             #     0.5,
    #             #     (0, 0, 255),
    #             #     1,
    #             #     cv2.LINE_AA,
    #             # )
    #             # frame_poses[pose.trunk_id] = keypoints.tolist()

    #             # # Draw the torso and trunk polygons
    #             # if pose.torso_polygon is not None:
    #             #     cv2.polylines(
    #             #         marked_frame,
    #             #         [pose.torso_polygon],
    #             #         isClosed=True,
    #             #         color=(75, 75, 75),
    #             #         thickness=2,
    #             #     )
    #             # if pose.trunk_polygon is not None:
    #             #     cv2.polylines(
    #             #         marked_frame,
    #             #         [pose.trunk_polygon],
    #             #         isClosed=True,
    #             #         color=(155, 155, 155),
    #             #         thickness=2,
    #             #     )

    #             # Copy the box region of the pose to the respective fighter frame
    #             x, y, w, h = bbox.xywh
    #             x1 = int(x - w / 2)
    #             x2 = int(x + w / 2)
    #             y1 = int(y - h / 2)
    #             y2 = int(y + h / 2)

    #             if pose.hmm_id == 0:
    #                 fighter_0_frame[y1:y2, x1:x2] = frame.pixels[y1:y2, x1:x2]
    #             if pose.hmm_id == 1:
    #                 fighter_1_frame[y1:y2, x1:x2] = frame.pixels[y1:y2, x1:x2]

    #         # poses_json_data[frame.idx] = {
    #         #     int(trunk_id): keypoints for trunk_id, keypoints in frame_poses.items()
    #         # }
    #         # cv2.imwrite(
    #         #     os.path.join(DIR_OUT, f"{self.clip_name}_frame{frame.idx}_trunk0.jpg"),
    #         #     fighter_0_frame,
    #         # )
    #         # cv2.imwrite(
    #         #     os.path.join(DIR_OUT, f"{self.clip_name}_frame{frame.idx}_trunk1.jpg"),
    #         #     fighter_1_frame,
    #         # )
    #         out_fighter_0.write(fighter_0_frame)
    #         out_fighter_1.write(fighter_1_frame)

    #     out_fighter_0.release()
    #     out_fighter_1.release()

    #     # with open(
    #     #     os.path.join(DIR_OUT, f"{self.clip_name}_poses_{POSE_TRACKER}.json"), "w"
    #     # ) as json_file:
    #     #     json.dump(poses_json_data, json_file, indent=4)

    #     return

    def output2(self, kpt=True):
        out_fighter_0 = cv2.VideoWriter(
            os.path.join(DIR_OUT, f"{self.clip_name}_fighter_op_0.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        out_fighter_1 = cv2.VideoWriter(
            os.path.join(DIR_OUT, f"{self.clip_name}_fighter_op_1.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        print("Exporting frames...")
        for frame in tqdm(self.frames):
            if kpt:
                out_fighter_0.write(frame.draw_keypoints(0))
                out_fighter_1.write(frame.draw_keypoints(1))
            else:
                out_fighter_0.write(frame.pixels_2fighters[0].copy())
                out_fighter_1.write(frame.pixels_2fighters[1].copy())

        out_fighter_0.release()
        out_fighter_1.release()
        return

    def export_npz(self):

        keypoints_data = {"positions_2d": {}}
        metadata = {
            "layout_name": "coco",
            "num_joints": 17,
            "keypoints_symmetry": [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
                [13, 14],
                [15, 16],
            ],
            "video_metadata": {},
        }

        # Correct mapping from BODY_25 to COCO keypoints
        body25_to_coco = [0, 15, 16, 17, 18, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14]
        coco_to_body25 = [
            0,
            -1,
            5,
            7,
            9,
            6,
            8,
            10,
            -1,
            11,
            13,
            15,
            12,
            14,
            16,
            1,
            2,
            3,
            4,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]

        # Process data for each fighter
        for fighter_idx in range(2):
            positions_2d = []

            for frame in self.frames:
                frame_keypoints = frame.pose_2fighters[fighter_idx][
                    body25_to_coco, :2
                ]  # Map and extract x, y coordinates
                positions_2d.append(frame_keypoints)

            positions_2d = np.array(positions_2d, dtype=np.float32)
            key = f"fighter_{fighter_idx}"
            keypoints_data["positions_2d"][key] = {"custom": [positions_2d]}
            metadata["video_metadata"][key] = {
                "w": 1280,  # Width of the video frame, change if different
                "h": 720,  # Height of the video frame, change if different
                "fps": 30,  # Frames per second of the video, change if different
            }

            # Save to .npz file
            np.savez_compressed(
                os.path.join(
                    "D:/Documents/devs/VideoPose3D/data",
                    f"{self.clip_name}_{fighter_idx}.npz",
                ),
                positions_2d=keypoints_data["positions_2d"],
                metadata=metadata,
            )
        return

    def split_2_fighters(self):

        print("Splitting fighters...")
        for frame in tqdm(self.frames):

            for bbox in frame.bboxes:
                # Copy the box region of the pose to the respective fighter frame
                x, y, w, h = bbox.xywh
                x1 = int(x - w / 2)
                x2 = int(x + w / 2)
                y1 = int(y - h / 2)
                y2 = int(y + h / 2)

                frame.pixels_2fighters[bbox.pose_yolo8.hmm_id][y1:y2, x1:x2] = (
                    frame.pixels[y1:y2, x1:x2]
                )
                frame.trunk_2fighters[bbox.pose_yolo8.hmm_id] = bbox.pose_yolo8.trunk_id

        return

    def generate_fighters_poses(self):
        track_history = defaultdict(lambda: [])
        drop_counting = defaultdict(lambda: 0)

        print("Tracking Poses...")
        for frame in tqdm(self.frames):
            track_history, drop_counting = frame.extract_fighter_pose_yolo8(
                track_history, drop_counting
            )

        return

    def bisection_trunk_color(self):
        trunk_colors = []

        # Collect all trunk colors from all frames and ensure consistent size
        for frame in self.frames:
            for bbox in frame.bboxes:
                trunk_hsv = bbox.pose_yolo8.trunk_hsv
                trunk_colors.append(trunk_hsv.flatten())

        # Convert to numpy array
        trunk_colors_np = np.array(trunk_colors)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=2, random_state=0).fit(trunk_colors_np)
        cluster_centers = kmeans.cluster_centers_

        # Iterate through each frame to assign trunk_id
        for frame in self.frames:
            if len(frame.bboxes) == 0:
                continue
            elif len(frame.bboxes) == 1:
                pose = frame.bboxes[0].pose_yolo8
                trunk_hsv = pose.trunk_hsv.flatten().reshape(1, -1)
                distances = np.linalg.norm(kmeans.cluster_centers_ - trunk_hsv, axis=1)
                pose.trunk_id = np.argmin(distances)
            else:
                # Calculate distances
                distances1 = np.linalg.norm(
                    cluster_centers
                    - frame.bboxes[0].pose_yolo8.trunk_hsv.flatten().reshape(1, -1),
                    axis=1,
                )
                distances2 = np.linalg.norm(
                    cluster_centers
                    - frame.bboxes[1].pose_yolo8.trunk_hsv.flatten().reshape(1, -1),
                    axis=1,
                )

                # Find the least-distant pair
                if distances1[0] + distances2[1] < distances1[1] + distances2[0]:
                    frame.bboxes[0].pose_yolo8.trunk_id = 0
                    frame.bboxes[1].pose_yolo8.trunk_id = 1
                else:
                    frame.bboxes[0].pose_yolo8.trunk_id = 1
                    frame.bboxes[1].pose_yolo8.trunk_id = 0

        return

    def drop_less_frequent_poses(self):
        track_id_counts = defaultdict(int)
        for frame in self.frames:
            for bbox in frame.bboxes:
                track_id_counts[bbox.pose_yolo8.track_id] += 1

        min_required_count = self.frame_count * 2 / len(track_id_counts)

        for frame in self.frames:
            frame.bboxes = [
                bbox
                for bbox in frame.bboxes
                if track_id_counts[bbox.pose_yolo8.track_id] >= min_required_count
            ]
            if len(frame.bboxes) > 2:

                frame.bboxes = sorted(
                    frame.bboxes, key=lambda b: -track_id_counts[b.pose_yolo8.track_id]
                )[:2]

        return

    def fill_missing_bbox(self):
        for i, frame in enumerate(self.frames):
            if len(frame.bboxes) < 2 and i > MIN_APPEARING_FRAMES:
                for trunk_id in [0, 1]:
                    if not any(
                        bbox.pose_yolo8.hmm_id == trunk_id for bbox in frame.bboxes
                    ):
                        prev_bbox = self.find_last_bbox(i, trunk_id)
                        next_bbox = self.find_next_bbox(i, trunk_id)
                        if prev_bbox is not None and next_bbox is not None:
                            interpolated_bbox = self.hull_bbox(
                                prev_bbox, next_bbox, frame
                            )
                            for j in range(
                                prev_bbox.frame.idx + 1, next_bbox.frame.idx
                            ):
                                self.frames[j].bboxes.append(
                                    interpolated_bbox.copy(True)
                                )
                        elif prev_bbox is not None:
                            for j in range(prev_bbox.frame.idx + 1, len(self.frames)):
                                if len(self.frames[j].bboxes) < 2:
                                    self.frames[j].bboxes.append(prev_bbox.copy(True))
                                else:
                                    break
                        elif next_bbox is not None:
                            for j in range(next_bbox.frame.idx - 1, -1, -1):
                                if len(self.frames[j].bboxes) < 2:
                                    self.frames[j].bboxes.append(next_bbox.copy(True))
                                else:
                                    break

        for frame in self.frames:
            frame.bboxes = sorted(frame.bboxes, key=lambda bbox: bbox.pose_yolo8.hmm_id)
        return

    def hull_bbox(self, bbox1, bbox2, frame):
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

    def find_last_bbox(self, current_idx, trunk_id):
        for i in range(current_idx - 1, -1, -1):
            for bbox in self.frames[i].bboxes:
                if bbox.pose_yolo8.trunk_id == trunk_id:
                    return bbox
        return None

    def find_next_bbox(self, current_idx, trunk_id):
        for i in range(current_idx + 1, len(self.frames)):
            for bbox in self.frames[i].bboxes:
                if bbox.pose_yolo8.trunk_id == trunk_id:
                    return bbox
        return None

    def hmm_track(self):
        # Prepare the observations and states
        observations = []
        state_sequence = []

        for frame in self.frames:
            if len(frame.bboxes) == 2:
                bbox1, bbox2 = frame.bboxes[0], frame.bboxes[1]
                xy1 = bbox1.xywh[:2]
                xy2 = bbox2.xywh[:2]
                trunk_id1 = bbox1.pose_yolo8.trunk_id
                trunk_id2 = bbox2.pose_yolo8.trunk_id
                observations.append([*xy1, trunk_id1])
                observations.append([*xy2, trunk_id2])
                state_sequence.append([trunk_id1, trunk_id2])  # Initial states
            elif len(frame.bboxes) == 1:
                bbox = frame.bboxes[0]
                xy = bbox.xywh[:2]
                trunk_id = bbox.pose_yolo8.trunk_id
                observations.append([*xy, trunk_id])
                state_sequence.append([trunk_id])  # Initial state for single bbox

        # Convert to numpy array
        observations = np.array(observations)

        # Create and fit the HMM model
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
        model.fit(observations)

        # Use Viterbi to find the most probable state sequence
        _, state_sequence = model.decode(observations, algorithm="viterbi")

        # Assign hmm_id based on the most probable state sequence
        idx = 0
        for frame in self.frames:
            if len(frame.bboxes) == 2:
                frame.bboxes[0].pose_yolo8.hmm_id = state_sequence[idx]
                frame.bboxes[1].pose_yolo8.hmm_id = state_sequence[idx + 1]
                idx += 2
            elif len(frame.bboxes) == 1:
                frame.bboxes[0].pose_yolo8.hmm_id = state_sequence[idx]
                idx += 1

        return

    def generate_2fighter_poses(self):
        opWrapper = op.WrapperPython()
        opWrapper.configure(OPENPOSE_PARAM)
        opWrapper.start()

        datum = op.Datum()

        print("Tracking 2 fighters separately with OpenPose...")
        for frame in tqdm(self.frames):
            frame.extract_fighter_pose_op(datum, opWrapper)

        return


def run_extract_fighters(fn_video):
    clip = Clip(fn_video)

    clip.generate_fighters_poses()

    clip.drop_less_frequent_poses()

    clip.bisection_trunk_color()
    clip.hmm_track()

    clip.fill_missing_bbox()  # Fill missing bounding boxes

    clip.split_2_fighters()

    clip.generate_2fighter_poses()

    clip.output2(kpt=False)
    clip.export_npz()

    cv2.destroyAllWindows()
    return

    # def direct_connection(self, items, bbox_dist_threshold):
    #     for frame_idx in range(self.frame_count - 1):
    #         current_items = getattr(self.frames[frame_idx], items)
    #         next_items = getattr(self.frames[frame_idx + 1], items)

    #         for ci in current_items:
    #             min_dist = float("inf")
    #             best_match = None
    #             for ni in next_items:
    #                 dist = Bbox.bbox_dist(ci, ni)
    #                 if dist < min_dist and dist <= bbox_dist_threshold:
    #                     min_dist = dist
    #                     best_match = ni

    #             if best_match:
    #                 ci.next = best_match
    #                 best_match.prev = ci
    #     return

    # def infer_connection(self, items, bbox_dist_threshold):
    #     for frame_idx in range(self.frame_count):
    #         current_items = getattr(self.frames[frame_idx], items)
    #         for ci in current_items:
    #             if ci.next is None:
    #                 for future_frame_idx in range(frame_idx + 1, self.frame_count):
    #                     future_items = getattr(self.frames[future_frame_idx], items)
    #                     for fi in future_items:
    #                         if fi.prev is None:
    #                             dist_threshold = (
    #                                 future_frame_idx - frame_idx
    #                             ) * bbox_dist_threshold
    #                             if Bbox.bbox_dist(ci, fi) <= dist_threshold:
    #                                 ci.next = fi
    #                                 fi.prev = ci
    #                                 break
    #                     if ci.next is not None:
    #                         break

    #     for frame_idx in range(self.frame_count - 1, 0, -1):
    #         current_items = getattr(self.frames[frame_idx], items)
    #         for ci in current_items:
    #             if ci.prev is None:
    #                 for past_frame_idx in range(frame_idx - 1, -1, -1):
    #                     past_items = getattr(self.frames[past_frame_idx], items)
    #                     for pi in past_items:
    #                         if pi.next is None:
    #                             dist_threshold = (
    #                                 frame_idx - past_frame_idx
    #                             ) * bbox_dist_threshold
    #                             if Bbox.bbox_dist(ci, pi) <= dist_threshold:
    #                                 ci.prev = pi
    #                                 pi.next = ci
    #                                 break
    #                     if ci.prev is not None:
    #                         break

    #     return

    # def interpolate_bbox(self, start_item, end_item):
    #     interpolated_items = []
    #     start_frame = start_item.frame.idx
    #     end_frame = end_item.frame.idx

    #     steps = end_frame - start_frame - 1
    #     if steps <= 0:
    #         return interpolated_items

    #     for i in range(1, steps + 1):
    #         ratio = i / (steps + 1)
    #         interpolated_xywh = (
    #             start_item.xywh[0] * (1 - ratio) + end_item.xywh[0] * ratio,
    #             start_item.xywh[1] * (1 - ratio) + end_item.xywh[1] * ratio,
    #             start_item.xywh[2] * (1 - ratio) + end_item.xywh[2] * ratio,
    #             start_item.xywh[3] * (1 - ratio) + end_item.xywh[3] * ratio,
    #         )
    #         confidence = (
    #             start_item.confidence * (1 - ratio) + end_item.confidence * ratio,
    #         )

    #         # do not use isinstance when subclass is involved
    #         if type(start_item) == Bbox:
    #             interpolated_items.append(
    #                 Bbox(
    #                     xywh=interpolated_xywh,
    #                     confidence=confidence,
    #                     frame=self.frames[start_frame + i],
    #                     is_interpolated=True,
    #                 )
    #             )
    #         elif type(start_item) == Contour:
    #             geometry = Contour.compute_geometry_from_xywh(interpolated_xywh)
    #             interpolated_items.append(
    #                 Contour(
    #                     geometry=geometry,
    #                     confidence=confidence,
    #                     frame=self.frames[start_frame + i],
    #                     is_interpolated=True,
    #                 )
    #             )

    #     return interpolated_items

    # def fill_connection(self, items):
    #     print("Filling missing segmentations...")
    #     for frame in tqdm(self.frames):
    #         item_list = getattr(frame, items)
    #         for item in item_list:
    #             if item.next and item.next.frame.idx != frame.idx + 1:
    #                 interpolated_items = self.interpolate_bbox(item, item.next)
    #                 for ii in interpolated_items:
    #                     getattr(self.frames[ii.frame.idx], items).append(ii)

    #             if item.prev and item.prev.frame.idx != frame.idx - 1:
    #                 interpolated_items = self.interpolate_bbox(item.prev, item)
    #                 for ii in interpolated_items:
    #                     getattr(self.frames[ii.frame.idx], items).append(ii)

    #     return

    # def generate_fighter_bboxes(self):
    #     min_area = MIN_AREA_RATIO * self.frame_width * self.frame_height

    #     print(f"Detecting human bbox by YOLO...")
    #     for frame in tqdm(self.frames):
    #         frame.extract_fighter_yolo(min_area)

    #     self.direct_connection("bboxes", BBOX_DIST_THRESHOLD)
    #     self.infer_connection("bboxes", BBOX_DIST_THRESHOLD)
    #     self.fill_connection("bboxes")

    #     print(f"Masking bbox at each frame...")
    #     for frame in tqdm(self.frames):
    #         mask_bbox = frame.mask_frame_with_bbox(frame.bboxes)
    #         frame.pixels = frame.crop_frame_with_mask(mask_bbox)

    #     return

    # def generate_fighter_contour(self):
    #     min_area = MIN_AREA_RATIO * self.frame_width * self.frame_height

    #     print(f"Detecting fighter contour by RCNN...")
    #     for frame in tqdm(self.frames):
    #         frame.extract_fighter_rcnn(min_area, SKIN_PCT_THRESHOLD)

    #     self.direct_connection("contours", BBOX_DIST_THRESHOLD)
    #     self.infer_connection("contours", BBOX_DIST_THRESHOLD)
    #     self.fill_connection("contours")

    #     print(f"Masking contour at each frame...")
    #     for frame in tqdm(self.frames):
    #         mask_contour = frame.mask_frame_with_contours(frame.contours)
    #         frame.pixels = frame.crop_frame_with_mask(mask_contour)

    #     return
