import torch
import torchvision
from ultralytics import YOLO

YOLO_THRESHOLD = 0.382
RCNN_THRESHOLD = 0.9
MIN_AREA_RATIO = 0.05
BBOX_DIST_THRESHOLD = 0.1
SKIN_PCT_THRESHOLD = 0.1
MASK_EXPAND_RATIO = 0.05
POSE_CONF_THRESHOLD = 0.618

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model
YOLO_BOX_MODEL = (
    torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval().to(DEVICE)
)
# YOLO_BOX_MODEL = YOLO("ultralytics/yolov8x.pt").to(DEVICE)
YOLO_BOX_MODEL.classes = [0]  # Set model to detect only people (class 0)

# Load Mask R-CNN model
MRCNN_MODEL = (
    torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    .eval()
    .to(DEVICE)
)

YOLO_POSE_MODEL = YOLO("weights/yolov8x-pose-p6.pt").to(DEVICE)
POSE_TRACKER = "bytetrack.yaml"  #'botsort.yaml
