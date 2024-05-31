import torch
import torchvision

YOLO_THRESHOLD = 0.382
RCNN_THRESHOLD = 0.9
MIN_AREA_RATIO = 0.05
BBOX_DIST_THRESHOLD = 0.1
SIGNIFICANT_DROP_RATIO = 0.618
MASK_EXPAND_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model
YOLO_MODEL = (
    torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval().to(DEVICE)
)
YOLO_MODEL.classes = [0]  # Set model to detect only people (class 0)

# Load Mask R-CNN model
MRCNN_MODEL = (
    torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    .eval()
    .to(DEVICE)
)
