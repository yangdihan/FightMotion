import numpy as np
import torch
import torchvision
from ultralytics import YOLO

YOLO_THRESHOLD = 0.382
RCNN_THRESHOLD = 0.9

MIN_AREA_RATIO = 0.05
BBOX_DIST_THRESHOLD = 0.1
SKIN_PCT_THRESHOLD = 0.2

POSE_CONF_THRESHOLD = 0.333
MIN_KEYPOINTS = 8
MIN_APPEARING_FRAMES = 6
MAX_MISSING_FRAMES = 24

MASK_EXPAND_RATIO = 0.05
SIZE_SQUARE_IMG = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load YOLOv5 model
# YOLO_BOX_MODEL = (
#     torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval().to(DEVICE)
# )
# # YOLO_BOX_MODEL = YOLO("ultralytics/yolov8x.pt").to(DEVICE)
# YOLO_BOX_MODEL.classes = [0]  # Set model to detect only people (class 0)

# # Load Mask R-CNN model
# MRCNN_MODEL = (
#     torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#     .eval()
#     .to(DEVICE)
# )

YOLO_POSE_MODEL = YOLO("weights/yolov8x-pose-p6.pt").to(DEVICE)
POSE_TRACKER = "botsort"  #'botsort.yaml

KEYPOINT_VISUAL = {
    "palette": [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ],
    "skeleton": [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ],
}

# Define color ranges in HSV
COLOR_RANGES = {
    "red": [
        (
            np.array([0, 75, 75], dtype=np.uint8),
            np.array([5, 255, 255], dtype=np.uint8),
        ),
        (
            np.array([165, 75, 75], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        ),
    ],
    "blue": [
        (
            np.array([85, 75, 75], dtype=np.uint8),
            np.array([125, 255, 255], dtype=np.uint8),
        )
    ],
    "yellow": [
        (
            np.array([20, 75, 75], dtype=np.uint8),
            np.array([55, 255, 255], dtype=np.uint8),
        )
    ],
    "green": [
        (
            np.array([55, 75, 75], dtype=np.uint8),
            np.array([85, 255, 255], dtype=np.uint8),
        )
    ],
    "purple": [
        (
            np.array([125, 75, 75], dtype=np.uint8),
            np.array([165, 255, 255], dtype=np.uint8),
        )
    ],
    "skin": [
        (
            np.array([5, 35, 35], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
    ],
    "white": [
        (
            np.array([0, 0, 128], dtype=np.uint8),
            np.array([180, 75, 255], dtype=np.uint8),
        )
    ],
    "black": [
        (np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 75], dtype=np.uint8))
    ],
}

# OpenPose parameters
# 1. BODY_25
# Description: Tracks 25 body keypoints including the major joints and extremities.
# Keypoints Tracked:
# 0-17: Head, neck, shoulders, elbows, wrists, hips, knees, ankles
# 18-23: Eyes, ears
# 24: Nose
# 2. COCO (BODY_18)
# Description: Tracks 18 body keypoints based on the COCO dataset.
# Keypoints Tracked:
# 0: Nose
# 1-2: Eyes (left, right)
# 3-4: Ears (left, right)
# 5-10: Shoulders, elbows, wrists (left, right)
# 11-16: Hips, knees, ankles (left, right)
# 3. MPI (BODY_15)
# Description: Tracks 15 body keypoints based on the MPII dataset.
# Keypoints Tracked:
# 0-14: Head, neck, shoulders, elbows, wrists, hips, knees, ankles
OPENPOSE_PARAM = {
    "model_folder": "D:\Apps\OpenPose\openpose_source\openpose\models",  # Update this path to your OpenPose models folder
    "model_pose": "BODY_25",
    # "keypoint_scale": 1,  # Scale the keypoints to the original image
    "number_people_max": 1,
    "tracking": 1,  # Enable tracking
    # "smooth": 1  # Enable smoothing
    # Disable hand keypoints
    # "hand": False,
    # # Disable face keypoints
    # "face": False,
    # # Enable body keypoints detection
    # "body": 1,
    # Utilize GPU
    "num_gpu": 1,
    # Start from GPU 0
    "num_gpu_start": 0,
    # # Enable rendering
    # "render_pose": 1,
}

POSE_PAIRS_25 = [
    (1, 8),
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (14, 19),
    (19, 20),
    (14, 21),
    (11, 22),
    (22, 23),
    (11, 24),
]

BODY25_TO_COCO = [0, 15, 16, 17, 18, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14]
COCO_to_BODY25 = [
    0,
    -1,
    5,
    7,
    9,
    6,
    8,
    10,
    -1,
    11,
    13,
    15,
    12,
    14,
    16,
    1,
    2,
    3,
    4,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
]

COCO_KEYPOINT_METADATA = {
    "layout_name": "coco",
    "num_joints": 17,
    "keypoints_symmetry": [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
    ],
    "video_metadata": {},
}
