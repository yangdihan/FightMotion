import numpy as np
import torch
import cv2
import pyopenpose as op

from constants import (
    YOLO_POSE_MODEL,
    OPENPOSE_PARAM,
    POSE_TRACKER,
    POSE_CONF_THRESHOLD,
    SKIN_PCT_THRESHOLD,
    BBOX_DIST_THRESHOLD,
    MIN_APPEARING_FRAMES,
    MAX_MISSING_FRAMES,
    MIN_KEYPOINTS,
    POSE_PAIRS_25,
)

from bbox import Bbox
from pose import Pose


class Frame:
    def __init__(self, idx, pixels) -> None:
        self.idx = idx
        self.pixels = pixels
        self.pixels_rgb = cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB)

        # datum = None

        self.bboxes = []
        # self.contours = []
        # self.poses = []

        self.pixels_2fighters = [np.zeros_like(self.pixels), np.zeros_like(self.pixels)]
        self.trunk_2fighters = [-1, -1]
        self.pose_2fighters = [np.zeros((25, 3)), np.zeros((25, 3))]
        # self.pose_2fighters_3d = [None, None]
        return

    def extract_fighter_pose_yolo8(self, track_history, drop_counting):

        result = YOLO_POSE_MODEL.track(
            self.pixels,
            persist=True,
            tracker=f"{POSE_TRACKER}.yaml",
            verbose=False,
            conf=POSE_CONF_THRESHOLD,
            # conf=0.1,
            iou=0.85,
            device="cuda",
        )[0]
        boxes = result.boxes.xywh.cpu()
        keypoints = result.keypoints.data
        track_ids = result.boxes.id

        if track_ids is None:
            track_ids = []
        else:
            track_ids = track_ids.int().cpu().tolist()

        diff = list(set(list(set(track_history.keys()))).difference(track_ids))
        for d in diff:
            if drop_counting[d] > MAX_MISSING_FRAMES:
                del drop_counting[d]
                del track_history[d]
            else:
                drop_counting[d] += 1

        for box, track_id, keypoint in zip(boxes, track_ids, keypoints):

            # check if bbox is big enough
            x, y, w, h = box
            if (
                w * h
                > self.pixels.shape[0] * self.pixels.shape[1] * BBOX_DIST_THRESHOLD
            ):
                bbox = Bbox(box, self, False)

                # Filter keypoints based on confidence
                keypoint_conf = keypoint[
                    ((keypoint[:, 0] > 0) | (keypoint[:, 1] > 0))
                    & (keypoint[:, 2] > POSE_CONF_THRESHOLD)
                ]

                # Filter out keypoints with less than MIN_KEYPOINTS points
                if keypoint_conf.shape[0] > MIN_KEYPOINTS:

                    track = track_history[track_id]
                    track.append(keypoint.unsqueeze(0))

                    if len(track) > MAX_MISSING_FRAMES:
                        track.pop(0)

                    # Only consider poses that have appeared for at least MIN_APPEARING_FRAMES frames
                    if len(track) >= MIN_APPEARING_FRAMES:
                        pose = Pose(torch.cat(track).cpu(), track_id, self, box)

                        if pose.pct_skin > SKIN_PCT_THRESHOLD:
                            # check if person is naked enough
                            # pose.classify_trunk_color()
                            # self.poses.append(pose)
                            bbox.pose_yolo8 = pose
                            self.bboxes.append(bbox)

        return track_history, drop_counting

    def extract_fighter_pose_op(self, datum, opWrapper):
        for i in [0, 1]:
            datum.cvInputData = self.pixels_2fighters[i]
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            keypoints = datum.poseKeypoints
            # print(keypoints)
            if keypoints is not None and len(keypoints) > 0:
                self.pose_2fighters[i] = keypoints[0]
        return

    def draw_keypoints(self, fighter_id):
        marked_frame = self.pixels_2fighters[fighter_id].copy()
        keypoints = self.pose_2fighters[fighter_id]

        if keypoints is not None and len(keypoints) > 0:
            # Draw the keypoints
            for i, keypoint in enumerate(keypoints):
                x, y, confidence = keypoint
                color = (
                    0,
                    255,
                    0,
                    int(255 * confidence),
                )  # RGBA color with alpha based on confidence
                overlay = marked_frame.copy()
                cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
                cv2.addWeighted(
                    overlay, confidence, marked_frame, 1 - confidence, 0, marked_frame
                )

            # Draw the skeleton
            for pair in POSE_PAIRS_25:
                partA, partB = pair
                if keypoints[partA][2] > 0.1 and keypoints[partB][2] > 0.1:
                    xA, yA, _ = keypoints[partA]
                    xB, yB, _ = keypoints[partB]
                    cv2.line(
                        marked_frame,
                        (int(xA), int(yA)),
                        (int(xB), int(yB)),
                        (0, 0, 0),
                        2,
                    )

        return marked_frame

    # def mask_frame_with_bbox(self, bboxes):
    #     mask = np.zeros_like(self.pixels)

    #     for bbox in bboxes:
    #         x, y, w, h = map(int, bbox.xywh)
    #         x, y, w, h = bbox.expand_bbox(MASK_EXPAND_RATIO)
    #         mask[y : y + h, x : x + w] = self.pixels[y : y + h, x : x + w]

    #     return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # def mask_frame_with_contours(self, contours):
    #     mask = np.zeros_like(self.pixels)

    #     for contour in contours:
    #         cv2.drawContours(
    #             mask, [contour.geometry], -1, (255, 255, 255), thickness=cv2.FILLED
    #         )

    #     return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # def crop_frame_with_mask(self, mask):
    #     # Apply the mask to the frame
    #     cropped_frame = cv2.bitwise_and(self.pixels, self.pixels, mask=mask)
    #     return cropped_frame

    # def extract_fighter_yolo(self, min_area):
    #     # Perform YOLO detection
    #     with torch.no_grad():
    #         results = YOLO_BOX_MODEL(self.pixels_rgb)

    #     for *box, conf, cls in results.xyxy[0].cpu().numpy():
    #         if cls == 0 and conf >= YOLO_THRESHOLD:
    #             # use customized heuristics to check if bbox is likely fighter
    #             x1, y1, x2, y2 = map(int, box)
    #             xywh = (x1, y1, x2 - x1, y2 - y1)

    #             bbox = Bbox(xywh=xywh, confidence=conf, frame=self)
    #             if bbox.area > min_area:
    #                 self.bboxes.append(bbox)

    #     return

    # def extract_fighter_rcnn(self, min_area, skin_pct_threshold):
    #     pil_img = (
    #         torch.tensor(self.pixels_rgb)
    #         .permute(2, 0, 1)
    #         .float()
    #         .div(255)
    #         .unsqueeze(0)
    #         .to(DEVICE)
    #     )

    #     # Perform Mask R-CNN detection
    #     with torch.no_grad():
    #         results = MRCNN_MODEL(pil_img)

    #     for idx in range(len(results[0]["masks"])):
    #         score = results[0]["scores"][idx].item()
    #         if score > RCNN_THRESHOLD:

    #             mask_rcnn = results[0]["masks"][idx, 0].mul(255).byte().cpu().numpy()
    #             contour_geoms, _ = cv2.findContours(
    #                 mask_rcnn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #             )

    #             # use customized heuristics to check if contour is likely fighter
    #             for contour_geom in contour_geoms:
    #                 contour = Contour(
    #                     geometry=contour_geom, confidence=score, frame=self
    #                 )

    #                 contour.estimate_skin_exposure()
    #                 # contour.detect_trunk_color()

    #                 if (
    #                     contour.area > min_area
    #                     and contour.pct_skin > skin_pct_threshold
    #                 ):
    #                     self.contours.append(contour)

    #     return
