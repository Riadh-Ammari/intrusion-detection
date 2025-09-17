import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

# Code 1: Pair YOLO detections with ground truth (compute IoU for matching)
def pair_yolo_detections_with_gt(yolo_model_path, image_path, gt_label_path, conf_threshold=0.2):
    # Load YOLO model
    model = YOLO(yolo_model_path)
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Run YOLO prediction
    results = model.predict(img, conf=conf_threshold)[0]
    
    # Extract predicted boxes (class, x_center, y_center, width, height, conf)
    pred_boxes = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        x_center = (x1 + x2) / 2 / img.shape[1]
        y_center = (y1 + y2) / 2 / img.shape[0]
        width = (x2 - x1) / img.shape[1]
        height = (y2 - y1) / img.shape[0]
        pred_boxes.append({'cls': int(cls), 'bbox': [x_center, y_center, width, height], 'conf': conf})
    
    # Load ground truth labels (class, x_center, y_center, width, height)
    gt_boxes = []
    with open(gt_label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            gt_boxes.append({'cls': cls, 'bbox': [x_center, y_center, width, height]})
    
    # Compute IoU for pairing
    def box_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1 - w1/2, x2 - w2/2)
        yi1 = max(y1 - h1/2, y2 - h2/2)
        xi2 = min(x1 + w1/2, x2 + w2/2)
        yi2 = min(y1 + h1/2, y2 + h2/2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    # Pair by highest IoU (per class)
    pairs = []
    for pred in pred_boxes:
        max_iou = 0.0
        best_gt = None
        for gt in gt_boxes:
            if pred['cls'] == gt['cls']:
                iou = box_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt
        pairs.append({'pred': pred, 'gt': best_gt, 'iou': max_iou})
    
    return pairs

# Example usage
yolo_model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_detection_v10_e30_20250815_1035\finetune\weights\best.pt"
image_path = Path(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\images\val\000009_jpg.rf.045ae48caa5e918e63c73ef2505a0447.jpg")
gt_label_path = Path(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\labels\val\000009_jpg.rf.045ae48caa5e918e63c73ef2505a0447.txt")
pairs = pair_yolo_detections_with_gt(yolo_model_path, image_path, gt_label_path)
print(pairs)