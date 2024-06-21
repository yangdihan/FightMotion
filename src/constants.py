import torch
import torchvision
from ultralytics import YOLO

YOLO_THRESHOLD = 0.382
RCNN_THRESHOLD = 0.9
MIN_AREA_RATIO = 0.05
BBOX_DIST_THRESHOLD = 0.1
SKIN_PCT_THRESHOLD = 0.1
MASK_EXPAND_RATIO = 0.05
POSE_CONF_THRESHOLD = 0.382
POSE_CONF_THRESHOLD = 0.382

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
