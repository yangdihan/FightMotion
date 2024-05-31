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


# OpenPose parameters
params = dict()
params["model_folder"] = (
    "D:\Apps\OpenPose\openpose_source\openpose\models"  # Update the path to your models folder
)
params["model_pose"] = "BODY_25"
params["hand"] = False  # Disable hand keypoints
params["face"] = False  # Disable face keypoints
params["body"] = 1  # Enable body keypoints detection
# params["tracking"] = 1  # Enable tracking
params["net_resolution"] = "-1x368"  # Use the appropriate resolution
params["num_gpu"] = 1  # Utilize GPU
params["num_gpu_start"] = 0  # Start from GPU 0
params["render_pose"] = 1  # Enable rendering

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# Class to store skeletal movements
class FighterPose:
    def __init__(self, frame_idx, pose_keypoints):
        self.frame_idx = frame_idx
        self.pose_keypoints = pose_keypoints


# Function to extract pose keypoints from a frame
def extract_pose_keypoints(frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData


# Load the video
video_path = "D:/Documents/devs/fight_motion/data/interim/output_video_contour.mp4"  # Update this to your video file path
cap = cv2.VideoCapture(video_path)

# Data structure to store keypoints
fighters_keypoints = []

# Video writer to save the output video
output_path = "D:/Documents/devs/fight_motion/data/interim/output_video_pose.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    30.0,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints, output_frame = extract_pose_keypoints(frame)
    # if keypoints is not None and len(keypoints) == 2:
    if keypoints is not None:
        fighters_keypoints.append(FighterPose(frame_idx, keypoints.tolist()))

        # Write the output frame with overlays to video
        out.write(output_frame)

    frame_idx += 1

cap.release()
out.release()

# Save the keypoints data
# with open("fighters_keypoints.json", "w") as f:
#     json.dump([fp.__dict__ for fp in fighters_keypoints], f)

# print("Keypoints extraction and video export completed.")
