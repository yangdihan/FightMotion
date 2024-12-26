import os

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


def show_frame_with_buttons(frame, fighters):
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


def first_frame_with_two_fighters(cap):
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
            for box in fighters:
                x, y, w, h = box
                x1 = int(x - w / 2)
                x2 = int(x + w / 2)
                y1 = int(y - h / 2)
                y2 = int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            accepted = show_frame_with_buttons(frame, fighters)
            if accepted:
                return frame_idx, fighters
        frame_idx += 1


def export_first_frame(frame_idx, fighters, cap, video_name):
    if frame_idx is not None and len(fighters) == 2:
        subfolder = os.path.join(DIR_INT, video_name)
        os.makedirs(subfolder, exist_ok=True)

        # Write fighter_1.txt and fighter_2.txt
        fighter_1_box = fighters[0]
        fighter_2_box = fighters[1]

        with open(os.path.join(subfolder, "fighter_1.txt"), "w") as f1:
            f1.write(
                f"{fighter_1_box[0]}, {fighter_1_box[1]}, {fighter_1_box[2]}, {fighter_1_box[3]}"
            )

        with open(os.path.join(subfolder, "fighter_2.txt"), "w") as f2:
            f2.write(
                f"{fighter_2_box[0]}, {fighter_2_box[1]}, {fighter_2_box[2]}, {fighter_2_box[3]}"
            )

        # Trim the video to start at frame_idx and remove audio
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        trimmed_video_path = os.path.join(subfolder, f"{video_name}_trimmed.mp4")
        out = cv2.VideoWriter(trimmed_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
    return


def main():
    for video_name in os.listdir(DIR_RAW):
        if video_name.endswith(".mp4"):
            video_path = os.path.join(DIR_RAW, video_name)
            # Open video
            cap = cv2.VideoCapture(video_path)

            frame_idx, fighters = first_frame_with_two_fighters(cap)
            export_first_frame(frame_idx, fighters, cap, video_name)

    cap.release()
    return


if __name__ == "__main__":
    main()
