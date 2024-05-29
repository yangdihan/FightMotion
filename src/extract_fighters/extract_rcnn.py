from tqdm import tqdm
import numpy as np
import torch
import torchvision
import cv2

from extract_fighters.constants import DEVICE
from extract_fighters.utils import bbox_dist, read_frame

# Load Mask R-CNN model
MRCNN_MODEL = (
    torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    .eval()
    .to(DEVICE)
)


def draw_contours_on_mask(contours, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    return mask


def extract_person_rcnn(frame, rcnn_model, min_confidence):
    pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = (
        torch.tensor(pil_img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
    )

    # Perform Mask R-CNN detection
    with torch.no_grad():
        results = rcnn_model(pil_img)

    contours_with_likelihood = []

    for idx in range(len(results[0]["masks"])):
        score = results[0]["scores"][idx].item()
        if score < min_confidence:
            continue

        mask_rcnn = results[0]["masks"][idx, 0].mul(255).byte().cpu().numpy()
        contours, _ = cv2.findContours(
            mask_rcnn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_with_likelihood.extend(contours)

    return contours_with_likelihood


def detect_skin(frame, contour):
    # Create a mask for the contour shape
    mask = draw_contours_on_mask([contour], frame.shape[:2])
    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Only check the upper 60% of the contour
    bottom_y = y + int(h * 0.618)

    # Mask out the bottom 40%
    mask[bottom_y:] = 0

    # Apply the mask to the frame
    cropped_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to HSV color space
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Calculate the percentage of skin area
    skin_area = cv2.countNonZero(skin_mask)
    total_area = cv2.countNonZero(mask)
    skin_percentage = skin_area / total_area if total_area > 0 else 0

    return skin_percentage


def evaluate_fighter_likelihood(frame, contour):
    # Calculate skin exposure in the upper 60% of the contour
    skin_percentage = detect_skin(frame, contour)

    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate bounding box area
    bbox_area = w * h

    # Normalize the bounding box area (you may need to adjust this normalization factor based on your video resolution)
    normalized_bbox_area = bbox_area / (frame.shape[0] * frame.shape[1])

    return (skin_percentage, normalized_bbox_area)


def extract_fighter_contour(frame, rcnn_model, min_confidence):
    contours = extract_person_rcnn(frame, rcnn_model, min_confidence)

    if not contours:
        return frame, [], []

    contours_with_likelihood = [
        (contour, evaluate_fighter_likelihood(frame, contour)) for contour in contours
    ]

    # Sort contours by combined likelihood and keep only the top two
    contours_with_likelihood.sort(
        key=lambda x: x[1][0] * 0.618 + x[1][1] * 0.382, reverse=True
    )
    top_contours = [c for c, _ in contours_with_likelihood][:2]

    return top_contours


def significant_drop(prev_count, curr_count, drop_ratio):
    return prev_count is not None and curr_count < prev_count * drop_ratio


def infer_missing_contours(contours_last, contours_this, bbox_dist_threshold):
    # Get bounding boxes for last and current contours
    bboxes_last = [cv2.boundingRect(contour) for contour in contours_last]
    bboxes_this = [cv2.boundingRect(contour) for contour in contours_this]

    paired_last = [False] * len(contours_last)
    paired_this = [False] * len(contours_this)

    for i, bbox_this in enumerate(bboxes_this):
        for j, bbox_last in enumerate(bboxes_last):
            if bbox_dist(bbox_this, bbox_last) <= bbox_dist_threshold:
                paired_this[i] = True
                paired_last[j] = True

    # Add unpaired contours from the last frame to the current frame
    for i, paired in enumerate(paired_last):
        if not paired:
            contours_this.append(contours_last[i])

    return contours_this


def main(video_stream, rcnn_threshold, significant_drop_ratio, bbox_dist_threshold):
    print(f"Detecting fighter contour by RCNN...")

    previous_non_blank_pixel_count = None
    top_contours_last = []

    for frame_idx in tqdm(range(video_stream.frame_count)):
        ret, frame = read_frame(video_stream.cap, frame_idx)
        if not ret:
            break

        top_contours = extract_fighter_contour(
            video_stream.frames[frame_idx].bbox_mask, MRCNN_MODEL, rcnn_threshold
        )

        mask_contour = draw_contours_on_mask(top_contours, frame.shape)
        non_blank_pixel_count = cv2.countNonZero(
            cv2.cvtColor(mask_contour, cv2.COLOR_BGR2GRAY)
        )

        if significant_drop(
            previous_non_blank_pixel_count,
            non_blank_pixel_count,
            significant_drop_ratio,
        ):
            top_contours = infer_missing_contours(
                top_contours_last, top_contours, bbox_dist_threshold
            )
            mask_contour = draw_contours_on_mask(top_contours, frame.shape)

        video_stream.frames[frame_idx].contour = top_contours
        video_stream.frames[frame_idx].contour_mask = mask_contour
        video_stream.frames[frame_idx].frame = cv2.bitwise_and(frame, mask_contour)

        previous_non_blank_pixel_count = non_blank_pixel_count
        top_contours_last = top_contours

    return video_stream
