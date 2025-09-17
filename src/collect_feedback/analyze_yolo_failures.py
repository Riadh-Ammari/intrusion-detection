import os
import numpy as np
from pathlib import Path
import shutil

def load_yolo_labels(label_path):
    """Load YOLO label file: class_id x_center y_center width height [confidence]"""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
        labels = [[int(l[0]), [float(x) for x in l[1:]]] for l in labels]
    return labels

def iou(box1, box2):
    """Calculate IoU between two bounding boxes [x_center, y_center, width, height]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def analyze_failures(pred_dir, gt_dir, img_dir, output_dir, conf_threshold=0.1, max_failures=500):
    """Analyze YOLO predictions to identify failures for all classes, prioritizing false positives"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    failures = []
    class_names = ['person', 'violence', 'weapon']
    
    for pred_file in Path(pred_dir).glob("*.txt"):
        img_name = pred_file.stem + ".jpg"
        if not os.path.exists(os.path.join(img_dir, img_name)):
            img_name = pred_file.stem + ".png"
            if not os.path.exists(os.path.join(img_dir, img_name)):
                img_name = pred_file.stem + ".jpeg"
                if not os.path.exists(os.path.join(img_dir, img_name)):
                    continue
        gt_file = Path(gt_dir) / pred_file.name
        pred_labels = load_yolo_labels(pred_file)
        gt_labels = load_yolo_labels(gt_file)
        
        # Prioritize false positives for all classes
        for class_id in range(3):  # person=0, violence=1, weapon=2
            class_pred = [l for l in pred_labels if l[0] == class_id]  # No conf threshold for false positives
            class_gt = [l for l in gt_labels if l[0] == class_id]
            if class_pred and not class_gt:
                failures.append((img_name, f"false_positive_{class_names[class_id]}", gt_labels, pred_labels))
        
        # Other failure types
        for class_id in range(3):
            class_pred = [l for l in pred_labels if l[0] == class_id and l[1][-1] > conf_threshold]
            class_gt = [l for l in gt_labels if l[0] == class_id]
            
            # False Negatives
            if class_gt and not class_pred:
                failures.append((img_name, f"false_negative_{class_names[class_id]}", gt_labels, pred_labels))
            
            # Incorrect Detections
            if class_pred and class_gt:
                for pred in class_pred:
                    matched = False
                    for gt in class_gt:
                        if pred[0] == gt[0] and iou(pred[1][:4], gt[1][:4]) > 0.5:
                            matched = True
                            break
                    if not matched:
                        failures.append((img_name, f"incorrect_detection_{class_names[class_id]}", gt_labels, pred_labels))
            
            # Low-Confidence Detections
            low_conf_preds = [l for l in pred_labels if l[0] == class_id and l[1][-1] <= conf_threshold]
            if low_conf_preds:
                failures.append((img_name, f"low_confidence_{class_names[class_id]}", gt_labels, pred_labels))
        
        if len(failures) >= max_failures:
            break
    
    # Save failure images and details
    for img_name, failure_type, gt_labels, pred_labels in failures:
        src_img = Path(img_dir) / img_name
        dst_img = Path(output_dir) / 'images' / img_name
        if src_img.exists():
            shutil.copy(src_img, dst_img)
        with open(Path(output_dir) / 'labels' / img_name.replace(Path(img_name).suffix, '.txt'), 'w') as f:
            f.write(f"Failure Type: {failure_type}\n")
            f.write(f"Ground Truth: {gt_labels}\n")
            f.write(f"Predictions: {pred_labels}\n")
    
    return failures

# Updated paths
pred_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\outputs\yolo_detection\val\labels"
gt_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\labels\val"
img_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\images\val"
output_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\rlhf_feedback"

failures = analyze_failures(pred_dir, gt_dir, img_dir, output_dir)
print(f"Found {len(failures)} failure cases. Saved to {output_dir}")