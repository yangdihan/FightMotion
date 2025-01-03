import os
from tqdm import tqdm
import torch
import cv2

import matplotlib.pyplot as plt
from ultralytics import YOLO
from bbox import Bbox
from pose import Pose  # Assuming these classes are defined in frame.py
from constants import (
    YOLO_POSE_MODEL,
    BBOX_DIST_THRESHOLD,
    POSE_CONF_THRESHOLD,
    MIN_KEYPOINTS,
    SKIN_PCT_THRESHOLD,
)
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # Ensure PIL is imported

DIR_RAW = "D:/Documents/devs/fight_motion/data/raw"
DIR_INT = "D:/Documents/devs/fight_motion/data/interim"


def show_frame_with_buttons(frame):
    # Create a Tkinter window
    root = tk.Tk()
    root.title("Frame Review")

    # Convert the frame to RGB format for displaying in Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Create a label to display the image
    img_label = tk.Label(root, image=imgtk)
    img_label.image = imgtk  # Keep a reference to the image
    img_label.pack()

    accept = False

    # Function to handle button clicks
    def on_accept():
        nonlocal accept  # Declare accept as nonlocal to modify the outer variable
        accept = True
        root.quit()
        root.destroy()

    def on_reject():
        root.quit()
        root.destroy()

    # Create Accept and Reject buttons
    accept_button = tk.Button(root, text="Accept", command=on_accept)
    reject_button = tk.Button(root, text="Reject", command=on_reject)
    accept_button.pack(side=tk.LEFT, padx=10, pady=10)
    reject_button.pack(side=tk.RIGHT, padx=10, pady=10)

    # Start the Tkinter event loop
    root.mainloop()

    return accept


def first_frame_with_two_fighters(video_name):
    video_path = os.path.join(DIR_RAW, video_name)
    # Open video
    cap = cv2.VideoCapture(video_path)
    # Iterate through frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Reached end of video without finding two fighters.")
            break

        # Run YOLO model on the frame
        results = YOLO_POSE_MODEL(frame)

        # Extract boxes and keypoints
        boxes = results[0].boxes.xywh  # [x, y, w, h]
        keypoints = results[0].keypoints

        fighters = []

        # Process each detected figure
        for box, keypoint in zip(boxes, keypoints):
            x, y, w, h = box
            if w * h > frame.shape[0] * frame.shape[1] * BBOX_DIST_THRESHOLD:
                keypoint = keypoint.data[0]
                keypoint_conf = keypoint[
                    ((keypoint[:, 0] > 0) | (keypoint[:, 1] > 0))
                    & (keypoint[:, 2] > POSE_CONF_THRESHOLD)
                ]

                # Check if the pose meets all conditions
                if keypoint_conf.shape[0] > MIN_KEYPOINTS:
                    pose = Pose(keypoint, None, frame, box)

                    if pose.pct_skin > SKIN_PCT_THRESHOLD:
                        # Detected a fighter
                        fighters.append(box)

        # Check if two fighters are detected
        if len(fighters) == 2:
            print(
                f"First frame with two fighters found. Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}"
            )

            # Plot the frame with the two fighters' bbox overlaid
            bbox_xyxy = []
            for box in fighters:
                x, y, w, h = box
                x1 = int(x - w / 2)
                x2 = int(x + w / 2)
                y1 = int(y - h / 2)
                y2 = int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                bbox_xyxy.append((x1, y1, x2, y2))

            accepted = show_frame_with_buttons(frame)
            if accepted:
                with open(
                    os.path.join(DIR_INT, video_name.replace(".mp4", ".txt")), "w"
                ) as f:
                    f.write(f"{frame_idx}\n{bbox_xyxy}")
                    break
        frame_idx += 1

    cap.release()
    return


def main():
    for video_name in tqdm(os.listdir(DIR_RAW)):
        if video_name.endswith(".mp4"):
            first_frame_with_two_fighters(video_name)
    return


if __name__ == "__main__":
    main()
