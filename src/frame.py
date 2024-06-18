import numpy as np
import torch
import cv2

from constants import (
    MASK_EXPAND_RATIO,
    YOLO_MODEL,
    # YOLO_THRESHOLD,
    MRCNN_MODEL,
    DEVICE,
    # RCNN_THRESHOLD,
)
from bbox import Bbox
from contour import Contour


class Frame:
    def __init__(self, idx, pixels) -> None:
        self.idx = idx
        self.pixels = pixels
        self.pixels_rgb = cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB)

        self.bboxes = []
        self.contours = []
        return

    def mask_frame_with_bbox(self, bboxes):
        mask = np.zeros_like(self.pixels)

        for bbox in bboxes:
            x, y, w, h = map(int, bbox.xywh)
            x, y, w, h = bbox.expand_bbox(MASK_EXPAND_RATIO)
            mask[y : y + h, x : x + w] = self.pixels[y : y + h, x : x + w]

        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    def mask_frame_with_contours(self, contours):
        mask = np.zeros_like(self.pixels)

        for contour in contours:
            cv2.drawContours(
                mask, [contour.geometry], -1, (255, 255, 255), thickness=cv2.FILLED
            )

        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    def crop_frame_with_mask(self, mask):
        # Apply the mask to the frame
        cropped_frame = cv2.bitwise_and(self.pixels, self.pixels, mask=mask)
        return cropped_frame

    # def mark_frame_with_bbox(self, bboxes):

    #     marked_frame = self.pixels.copy()
    #     for bbox in bboxes:
    #         x, y, w, h = map(int, bbox.xywh)
    #         cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     return marked_frame

    def extract_fighter_yolo(self, yolo_conf_threshold, min_area):
        # Perform YOLO detection
        with torch.no_grad():
            results = YOLO_MODEL(self.pixels_rgb)

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if cls == 0 and conf >= yolo_conf_threshold:
                # use customized heuristics to check if bbox is likely fighter
                x1, y1, x2, y2 = map(int, box)
                xywh = (x1, y1, x2 - x1, y2 - y1)

                bbox = Bbox(xywh=xywh, confidence=conf, frame=self)
                if bbox.area > min_area:
                    self.bboxes.append(bbox)

        return

    def extract_fighter_rcnn(self, rcnn_conf_threshold, min_area, skin_pct_threshold):
        pil_img = (
            torch.tensor(self.pixels_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # Perform Mask R-CNN detection
        with torch.no_grad():
            results = MRCNN_MODEL(pil_img)

        for idx in range(len(results[0]["masks"])):
            score = results[0]["scores"][idx].item()
            if score > rcnn_conf_threshold:

                mask_rcnn = results[0]["masks"][idx, 0].mul(255).byte().cpu().numpy()
                contour_geoms, _ = cv2.findContours(
                    mask_rcnn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # use customized heuristics to check if contour is likely fighter
                for contour_geom in contour_geoms:
                    contour = Contour(
                        geometry=contour_geom, confidence=score, frame=self
                    )

                    contour.estimate_skin_exposure()
                    # contour.detect_trunk_color()

                    if (
                        contour.area > min_area
                        and contour.pct_skin > skin_pct_threshold
                    ):
                        self.contours.append(contour)

        return
