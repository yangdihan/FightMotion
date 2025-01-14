import os
import gc
import json  # Import json for storing contours
from tqdm import tqdm
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import load_video_frames
from constants import YOLO_POSE_MODEL, POSE_CONF_THRESHOLD

DIR_RAW = "D:/Documents/devs/fight_motion/data/raw"
DIR_INT = "D:/Documents/devs/fight_motion/data/interim"
DIR_SAM = "D:/Documents/devs/fight_motion/sam2-main"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define colors for the masks
colors = [(255, 0, 0), (0, 0, 255)]  # Red and Blue

def load_prompts(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    frame_idx = int(lines[0].strip())
    bbox_list = eval(lines[1].strip())
    return frame_idx, bbox_list

def extract_fighter_masks(video_path, txt_path, output_video_path, mask_type='color'):
    checkpoint = os.path.join(DIR_SAM, "checkpoints/sam2.1_hiera_small.pt")
    model_cfg = os.path.join(DIR_SAM, "sam2/configs/sam2.1/sam2.1_hiera_s.yaml")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    state = predictor.init_state(video_path, offload_video_to_cpu=True)
    frame0, prompts = load_prompts(txt_path)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    loaded_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        loaded_frames.append(frame)
    cap.release()

    height, width = loaded_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    fighter_masks = ([], [])
    contours_data = []  # To store contours for each frame

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for obj_id, bbox in enumerate(prompts):
            _, _, _ = predictor.add_new_points_or_box(state, frame_idx=frame0, obj_id=obj_id, box=bbox)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                fighter_masks[obj_id].append(mask)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=frame0, reverse=True):
            if frame_idx == frame0:
                continue
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                fighter_masks[obj_id].insert(0, mask)

        for frame_idx in tqdm(range(len(loaded_frames))):
            img = loaded_frames[frame_idx].copy()
            frame_contours = []  # Store contours for this frame
            if mask_type == 'color':
                for obj_id, masks in enumerate(fighter_masks):
                    if frame_idx < len(masks):
                        mask = masks[frame_idx]
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = colors[obj_id]
                        img = cv2.addWeighted(img, 1, mask_img, 0.5, 0)
                        # Find contours
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        frame_contours.append(contours)
            elif mask_type == 'boolean':
                union_mask = np.zeros((height, width), dtype=np.uint8)  # Use uint8 for compatibility with OpenCV
                for obj_id, masks in enumerate(fighter_masks):
                    if frame_idx < len(masks):
                        mask = masks[frame_idx]
                        union_mask = np.logical_or(union_mask, mask).astype(np.uint8)  # Union of masks

                # Find contours of the union mask
                contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Combine all contour points into a single array
                all_points = np.vstack(contours)
                
                # Calculate the convex hull for all points combined
                if len(all_points) > 0:
                    hull = cv2.convexHull(all_points)
                
                    # Create a mask for the convex hull
                    hull_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(hull_mask, [hull], -1, (1), thickness=cv2.FILLED)

                    # Apply the convex hull mask to the image
                    img[hull_mask == 0] = 0
                # Find contours
                contours, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_contours.append(contours)

            contours_data.append(frame_contours)
            out.write(img)

            contours_data.append(frame_contours)
            out.write(img)

    out.release()

    # Save contours to a JSON file
    contours_file_path = output_video_path.replace('.mp4', '_contours.json')
    with open(contours_file_path, 'w') as f:
        json.dump(contours_data, f, cls=NumpyEncoder)

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

for video_name in os.listdir(DIR_RAW):
    mask_type = 'boolean'
    if video_name.endswith(".mp4") and video_name == 'aldo_holloway_2.mp4':
        txt_path = os.path.join(DIR_INT, video_name.replace(".mp4", ".txt"))
        video_path = os.path.join(DIR_RAW, video_name)
        output_video_path = os.path.join(DIR_INT, video_name.replace(".mp4", f"_{mask_type}.mp4"))
        print("extracting fighter masks for", video_name)
        extract_fighter_masks(video_path, txt_path, output_video_path, mask_type=mask_type)