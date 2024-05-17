import numpy as np
from scipy.interpolate import interp1d
import torch
import torchvision
import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval().to(DEVICE)
YOLO_MODEL.classes = [0]  # Set model to detect only people (class 0)

# Load Mask R-CNN model
# mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().cuda()

def extract_person_yolo(frame, yolo_model, threshold, min_area):
    # Convert frame to RGB (YOLO model expects RGB images)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform detection
    results = yolo_model(img)

    # Draw bounding boxes and labels on the frame
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if cls == 0 and conf >= threshold:  # class 0 is person
            x1, y1, x2, y2 = map(int, box)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area >= min_area:
                label = f'{yolo_model.names[int(cls)]} {conf:.2f}'
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame


def main(input_video_path, output_video_path, yolo_threshold, maskrcnn_threshold, min_area_ratio):
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    min_area = min_area_ratio * frame_width * frame_height

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        extract_person_yolo(frame, YOLO_MODEL, yolo_threshold, min_area)

        # Write the frame to the output video file
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    input_video_path = 'D:/Documents/devs/fight_motion/data/raw/aldo_holloway_single_angle.mp4'
    output_video_path = 'D:/Documents/devs/fight_motion/data/interim/aldo_holloway_yolo_test.mp4'
    main(input_video_path, output_video_path, 0.3, 0.5, 0.05)
