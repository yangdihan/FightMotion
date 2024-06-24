import os
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import torch
import cv2


from constants import (
    POSE_TRACKER,
)
from frame import Frame


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

                # Draw the torso and trunk polygons
                if pose.torso_polygon is not None:
                    cv2.polylines(
                        marked_frame,
                        [pose.torso_polygon],
                        isClosed=True,
                        color=(75, 75, 75),
                        thickness=2,
                    )
                if pose.trunk_polygon is not None:
                    cv2.polylines(
                        marked_frame,
                        [pose.trunk_polygon],
                        isClosed=True,
                        color=(155, 155, 155),
                        thickness=2,
                    )

                frame_poses.append(
                    {
                        "id": pose.track_id,
                        "keypoints": keypoints.tolist(),
                        "pct_skin": pose.pct_skin,
                        "trunk": pose.trunk_id,
                        # "trunk_color": pose.trunk_color,
                    }
                )

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

    def generate_fighter_poses(self):
        track_history = defaultdict(lambda: [])
        drop_counting = defaultdict(lambda: 0)

        print("Tracking Poses...")
        for frame in tqdm(self.frames):
            track_history, drop_counting = frame.extract_fighter_pose(
                track_history, drop_counting
            )

        return

    def bisection_trunk_color(self):
        trunk_colors = []

        # Collect all trunk colors from all frames and ensure consistent size
        for frame in self.frames:
            for pose in frame.poses:
                trunk_img = pose.trunk_img.cpu().numpy()
                if (
                    trunk_img.shape[1] != 8 or trunk_img.shape[2] != 8
                ):  # Resize to 8x8 if necessary
                    trunk_img = cv2.resize(
                        trunk_img.transpose(1, 2, 0), (8, 8)
                    ).transpose(2, 0, 1)
                trunk_colors.append(trunk_img.flatten())

        # Convert to numpy array
        trunk_colors_np = np.array(trunk_colors)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=2, random_state=0).fit(trunk_colors_np)

        # # Visualize the cluster centers
        # cluster_centers = kmeans.cluster_centers_.reshape(-1, 3, 8, 8)
        # cluster_centers_rgb = []
        # for center in cluster_centers:
        #     center_hsv = (center * 255).astype(np.uint8).transpose(1, 2, 0)
        #     center_rgb = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2RGB)
        #     cluster_centers_rgb.append(center_rgb)

        # import matplotlib.pyplot as plt
        # # Display the cluster centers
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # for ax, center_rgb in zip(axes, cluster_centers_rgb):
        #     ax.imshow(center_rgb)
        #     ax.axis('off')
        # plt.show()
        # raise ValueError('debug')

        # Update tracker_id based on the closest color cluster
        for frame in self.frames:
            for pose in frame.poses:
                trunk_img = pose.trunk_img.cpu().numpy()
                if (
                    trunk_img.shape[1] != 8 or trunk_img.shape[2] != 8
                ):  # Resize to 8x8 if necessary
                    trunk_img = cv2.resize(
                        trunk_img.transpose(1, 2, 0), (8, 8)
                    ).transpose(2, 0, 1)
                pose_color_np = trunk_img.flatten().reshape(1, -1)
                distances = kmeans.transform(pose_color_np)
                closest_cluster = np.argmin(distances)
                pose.trunk_id = closest_cluster

        return


def run_extract_fighters(fn_video):
    clip = Clip(fn_video)

    # clip.generate_fighter_bboxes()
    # clip.generate_fighter_contour()
    clip.generate_fighter_poses()

    clip.bisection_trunk_color()

    clip.cap.release()
    clip.output()
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
