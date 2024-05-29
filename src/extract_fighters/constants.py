import numpy as np
import torch

YOLO_THRESHOLD = 0.382
RCNN_THRESHOLD = 0.9
MIN_AREA_RATIO = 0.05
BBOX_DIST_THRESHOLD = 0.1
SIGNIFICANT_DROP_RATIO = 0.618
MASK_EXPAND_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
