# Import necessary libraries
import os
import gc
import json  # Import json for storing contours
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np  # Import numpy for numerical operations
import cv2  # Import OpenCV for image processing
import torch  # Import PyTorch for deep learning
from sam2.build_sam import (
    build_sam2_video_predictor,
)  # Import function to build SAM predictor
from sam2.utils.misc import load_video_frames  # Import function to load video frames
from constants import (
    YOLO_POSE_MODEL,
    POSE_CONF_THRESHOLD,
)  # Import constants for YOLO model

# Define directories for raw and interim data
DIR_RAW = "D:/Documents/devs/fight_motion/data/raw"
DIR_INT = "D:/Documents/devs/fight_motion/data/interim"
DIR_SAM = "D:/Documents/devs/fight_motion/sam2-main"

# Check if CUDA is available and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU if available
else:
    device = torch.device("cpu")  # Use CPU if GPU is not available

# Define colors for the masks
colors = [(255, 0, 0), (0, 0, 255)]  # Red and Blue


# Function to load prompts from a text file
def load_prompts(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()  # Read all lines from the file
    frame_idx = int(lines[0].strip())  # Get the frame index
    bbox_list = eval(lines[1].strip())  # Evaluate the bounding box list
    return frame_idx, bbox_list  # Return frame index and bounding box list


# Function to squeeze contours into a convex hull
def squeeze_contours(contours):
    # Combine all contour points into a single array
    all_points = np.vstack(contours)
    # Calculate the convex hull for all points combined
    hull = cv2.convexHull(all_points)
    # Use the convex hull to find contours
    contours = [contour[0] for contour in hull]  # Remove the extra layer of list
    return contours, hull  # Return contours and hull


# Function to extract fighter masks from a video
def extract_fighter_masks(
    video_path, txt_path, output_video_path, output_contours_path, mask_type="color"
):
    # Define paths for model checkpoint and configuration
    checkpoint = os.path.join(DIR_SAM, "checkpoints/sam2.1_hiera_small.pt")
    model_cfg = os.path.join(DIR_SAM, "sam2/configs/sam2.1/sam2.1_hiera_s.yaml")
    # Build the SAM predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    # Initialize the predictor state
    state = predictor.init_state(video_path, offload_video_to_cpu=True)
    frame0, prompts = load_prompts(txt_path)  # Load prompts from the text file

    # Capture video frames
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    loaded_frames = []  # List to store loaded frames
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Break the loop if no frame is returned
        loaded_frames.append(frame)  # Append the frame to the list
    cap.release()  # Release the video capture object

    # Get the height and width of the first frame
    height, width = loaded_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define codec for video writing
    out = cv2.VideoWriter(
        output_video_path, fourcc, frame_rate, (width, height)
    )  # Create VideoWriter object

    fighter_masks = ([], [])  # Initialize fighter masks
    contours_data = []  # To store contours for each frame

    # Perform inference with the predictor
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for obj_id, bbox in enumerate(prompts):
            _, _, _ = predictor.add_new_points_or_box(
                state, frame_idx=frame0, obj_id=obj_id, box=bbox
            )

        # Propagate masks in the video
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()  # Convert mask to numpy array
                mask = mask > 0.0  # Threshold the mask
                fighter_masks[obj_id].append(
                    mask
                )  # Append mask to the corresponding fighter

        # Propagate masks in reverse order
        for frame_idx, object_ids, masks in predictor.propagate_in_video(
            state, start_frame_idx=frame0, reverse=True
        ):
            if frame_idx == frame0:
                continue  # Skip the first frame
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()  # Convert mask to numpy array
                mask = mask > 0.0  # Threshold the mask
                fighter_masks[obj_id].insert(0, mask)  # Insert mask at the beginning

    # Process each frame for visualization
    for frame_idx in tqdm(range(len(loaded_frames))):
        img = loaded_frames[frame_idx].copy()  # Copy the current frame

        frame_contours = [None, None, None]  # Initialize contours for the frame

        for obj_id, masks in enumerate(fighter_masks):
            mask = masks[frame_idx]  # Get the mask for the current frame
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )  # Find contours
            contours, _ = squeeze_contours(contours)  # Squeeze contours
            frame_contours[obj_id] = contours  # Store contours for the object

            if mask_type == "color":
                mask_img = np.zeros(
                    (height, width, 3), np.uint8
                )  # Create an empty mask image
                mask_img[mask] = colors[obj_id]  # Color the mask
                img = cv2.addWeighted(
                    img, 1, mask_img, 0.5, 0
                )  # Blend the mask with the image

        # Find contours of the union mask
        contours, hull = squeeze_contours(frame_contours[0] + frame_contours[1])
        frame_contours[2] = contours  # Store union contours

        if mask_type == "boolean":
            # Create a mask for the convex hull
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(
                hull_mask, [hull], -1, (1), thickness=cv2.FILLED
            )  # Draw the hull
            # Apply the convex hull mask to the image
            img[hull_mask == 0] = 0

        contours_data.append(frame_contours)  # Append contours data for the frame
        out.write(img)  # Write the processed image to the output video

    out.release()  # Release the video writer
    # Save contours to a JSON file
    with open(output_contours_path, "w") as f:
        json.dump(contours_data, f, cls=NumpyEncoder)  # Dump contours data to JSON

    del predictor, state  # Delete predictor and state to free memory
    gc.collect()  # Run garbage collection
    torch.clear_autocast_cache()  # Clear autocast cache
    torch.cuda.empty_cache()  # Clear CUDA cache
    return  # Return from the function


# Custom JSON encoder for numpy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        return json.JSONEncoder.default(self, obj)  # Default behavior for other types


# Process each video in the raw directory
for video_name in os.listdir(DIR_RAW):
    mask_type = "boolean"  # Define mask type
    if video_name.endswith(".mp4") and video_name == "aldo_holloway_2.mp4":
        # if video_name.endswith(".mp4"):
        txt_path = os.path.join(
            DIR_INT, video_name.replace(".mp4", ".txt")
        )  # Define path for text file
        video_path = os.path.join(DIR_RAW, video_name)  # Define path for video
        output_video_path = os.path.join(
            DIR_INT, video_name.replace(".mp4", f"_{mask_type}.mp4")
        )  # Define output video path
        output_contours_path = os.path.join(
            DIR_INT, video_name.replace(".mp4", "_contours.json")
        )  # Define output contours path
        print("extracting fighter masks for", video_name)  # Print message
        extract_fighter_masks(
            video_path,
            txt_path,
            output_video_path,
            output_contours_path,
            mask_type=mask_type,
        )  # Call function to extract masks
