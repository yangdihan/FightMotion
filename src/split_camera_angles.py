import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

from constants import *

THRES_FRAME_SIMILARITY = 0.7

# def frame_similarity(frame1, frame2):
#     """Calculate the structural similarity between two frames."""
#     frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     similarity = ssim(frame1_gray, frame2_gray)
#     return similarity

def frame_similarity(frame1, frame2):
    """Calculate frame similarity using absolute difference and count of non-zero pixels using CUDA."""
    # Calculate the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale to ensure single channel for counting non-zero pixels
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Count non-zero pixels in the difference image
    non_zero_count = cv2.countNonZero(diff_gray)

    # Normalize the count of non-zero pixels by the total number of pixels to get a change fraction
    total_pixels = frame1.size
    change_fraction = non_zero_count / total_pixels

    # Convert change fraction to a similarity score (1 - change_fraction)
    similarity_score = 1 - change_fraction

    return similarity_score


def save_clip(start_frame, end_frame, video_path, output_folder, cap, frame_width, frame_height, fps):
    """Save the video clip from start_frame to end_frame if it is at least 1 second long."""
    frame_count = end_frame - start_frame
    if frame_count < fps:  # Check if the segment length is less than 1 second
        print(f"Skipping clip from Frame {start_frame} to {end_frame} - less than 1 second long.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_path = os.path.join(output_folder, f'clip_{start_frame}_{end_frame}.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()
    print(f"Saved clip: {output_path}")

def main(video_path, output_folder):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    frame_index = 0
    start_frame = 0  # Start of new segment
    while True:
        # Read next frame
        ret, curr_frame = cap.read()
        if not ret:
            # Save the last segment
            if frame_index > start_frame:
                save_clip(start_frame, frame_index, video_path, output_folder, cap, frame_width, frame_height, fps)
                break
        
        # Calculate similarity
        similarity = frame_similarity(prev_frame, curr_frame)

        # If similarity is below a threshold, it might be a camera change
        if similarity < THRES_FRAME_SIMILARITY:  # This threshold might need adjustment
            print(f"Camera angle change detected between Frame {frame_index} and Frame {frame_index + 1}")
            # Save the previous segment
            save_clip(start_frame, frame_index, video_path, output_folder, cap, frame_width, frame_height, fps)
            # Start a new segment
            start_frame = frame_index + 1
        
        # Update previous frame and index
        prev_frame = curr_frame
        frame_index += 1
    
    cap.release()

if __name__ == "__main__":
    video_path = os.path.join(DATA_DIR_RAW, 'tj_2_angles.mp4')  # Update this with the path to your video
    output_folder = DATA_DIR_INT  # Folder to save clips
    main(video_path, output_folder)

    # if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    #     print("CUDA is not available in your OpenCV installation")
    # else:
    #     print('yea')
