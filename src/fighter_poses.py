import os
import sys
import json
from tqdm import tqdm

# from sys import platform
# import argparse
# import cv2
# import pyopenpose as op


import numpy as np
import cv2
import pyopenpose as op

video_path = "D:/Documents/devs/fight_motion/data/interim/aldo_holloway_angle1_fighter_0.mp4"  # Update this path to your video file

cap = cv2.VideoCapture(video_path)


# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(
    "D:/Documents/devs/fight_motion/data/interim/aldo_holloway_angle1_fighter_0_openpose.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

# OpenPose parameters
params = {
    "model_folder": "D:\Apps\OpenPose\openpose_source\openpose\models",  # Update this path to your OpenPose models folder
    # "keypoint_scale": 1,  # Scale the keypoints to the original image
    "tracking": 1,  # Enable tracking
    "number_people_max": 1,
    # "smooth": 1  # Enable smoothing
}
params["model_pose"] = "MPI"  # body 15
params["hand"] = False  # Disable hand keypoints
params["face"] = False  # Disable face keypoints
params["body"] = 1  # Enable body keypoints detection
# params["tracking"] = 1  # Enable tracking
# params["net_resolution"] = "-1x368"  # Use the appropriate resolution
params["num_gpu"] = 1  # Utilize GPU
params["num_gpu_start"] = 0  # Start from GPU 0
# params["render_pose"] = 1  # Enable rendering


# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# Function to extract pose keypoints
def extract_pose(frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum


# Function to draw keypoints on the frame
def draw_keypoints(frame, keypoints):
    for person in keypoints:
        for i in range(len(person)):
            x, y, confidence = person[i]
            if confidence > 0.1:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    return frame


# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract 2D pose keypoints
    datum = extract_pose(frame)
    keypoints_2d = datum.poseKeypoints
    print("frame")
    # Draw keypoints on the frame
    if keypoints_2d is not None:
        frame = draw_keypoints(frame, keypoints_2d)

    # Write the frame with keypoints
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
