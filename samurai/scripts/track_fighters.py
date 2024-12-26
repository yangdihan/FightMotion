import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

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


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError(
            "Invalid video_path format. Should be .mp4 or a directory of jpg frames."
        )


def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    frame0, prompts = load_txt(args.txt_path)

    frame_rate = 30
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted(
                [
                    osp.join(args.video_path, f)
                    for f in os.listdir(args.video_path)
                    if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))
                ]
            )
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    # Initialize mask storage for both fighters
    fighter_masks = ([], [])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for obj_id in range(2):  # Assuming two fighters
            state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
            bbox = prompts[obj_id]
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
                img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

            out.write(img)

        if args.save_to_video:
            out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", required=True, help="Input video path or directory of frames."
    )
    parser.add_argument(
        "--txt_path", required=True, help="Path to ground truth text file."
    )
    parser.add_argument(
        "--model_path",
        default="sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--video_output_path", default="demo.mp4", help="Path to save the output video."
    )
    parser.add_argument(
        "--save_to_video", default=True, help="Save results to a video."
    )
    args = parser.parse_args()
    main(args)
