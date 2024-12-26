import argparse
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import torch
import gc
import sys

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

DIR_RAW = "D:/Documents/devs/fight_motion/data/raw"
DIR_INT = "D:/Documents/devs/fight_motion/data/interim"

color = [(255, 0, 0), (0, 0, 255)]


def load_txt(gt_path):
    with open(gt_path, "r") as f:
        frame0 = int(f.readline().strip())  # Read the frame index from the first line
        bbox_str = f.readline().strip()  # Read the bounding boxes from the second line

    # Convert bounding box strings to numpy arrays
    bbox_list = eval(
        bbox_str
    )  # Use eval to convert the string representation of the list to an actual list
    bbox1, bbox2 = tuple(bbox_list)  # Unpack the bounding boxes

    # Create prompts tuple with bounding boxes
    prompts = (bbox1, bbox2)

    return frame0, prompts


def extract_fighter_masks(video_path, txt_path, output_video_path):
    # model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(
        "configs/samurai/sam2.1_hiera_s.yaml",
        "sam2/checkpoints/sam2.1_hiera_small.pt",
        device="cuda:0",
    )

    frame0, prompts = load_txt(txt_path)

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

    # Initialize mask storage for both fighters
    fighter_masks = ([], [])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for obj_id, bbox in enumerate(prompts):  # Assuming two fighters
            state = predictor.init_state(video_path, offload_video_to_cpu=True)
            # bbox = prompts[obj_id]
            _, _, _ = predictor.add_new_points_or_box(
                state, box=bbox, frame_idx=frame0, obj_id=obj_id
            )

            # Propagate tracking through video for each fighter
            print("extracking fighter", obj_id, "...")
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                for mask in masks:
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    fighter_masks[obj_id].append(mask)

        # Overlay masks on the original video
        print("exporting video...")
        for frame_idx in tqdm(range(len(loaded_frames))):
            img = loaded_frames[frame_idx]
            for obj_id, masks in enumerate(fighter_masks):
                mask = masks[frame_idx]
                mask_img = np.zeros((height, width, 3), np.uint8)
                mask_img[mask] = color[obj_id]
                img = cv2.addWeighted(
                    img, 1, mask_img, 0.8, 0
                )  # Increased opacity to make color more solid

            out.write(img)

        out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    return


def main():
    for video_name in tqdm(os.listdir(DIR_RAW)):
        if video_name.endswith(".mp4"):
            txt_path = os.path.join(DIR_INT, video_name.replace(".mp4", ".txt"))
            video_path = os.path.join(DIR_RAW, video_name)
            output_video_path = os.path.join(
                DIR_INT, video_name.replace(".mp4", "_masks.mp4")
            )

            extract_fighter_masks(video_path, txt_path, output_video_path)
    return


if __name__ == "__main__":
    main()


# def determine_model_cfg(model_path):
#     if "large" in model_path:
#         return "configs/samurai/sam2.1_hiera_l.yaml"
#     elif "base_plus" in model_path:
#         return "configs/samurai/sam2.1_hiera_b+.yaml"
#     elif "small" in model_path:
#         return "configs/samurai/sam2.1_hiera_s.yaml"
#     elif "tiny" in model_path:
#         return "configs/samurai/sam2.1_hiera_t.yaml"
#     else:
#         raise ValueError("Unknown model size in path!")


# def prepare_frames_or_path(video_path):
#     if video_path.endswith(".mp4") or osp.isdir(video_path):
#         return video_path
#     else:
#         raise ValueError(
#             "Invalid video_path format. Should be .mp4 or a directory of jpg frames."
#         )
