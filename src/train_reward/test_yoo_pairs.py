import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json
from torchvision import transforms
import os
from fine_tuning.reward_model import RewardModel

# Paths
yolo_model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_detection_v10_e30_20250815_1035\finetune\weights\best.pt"
reward_model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\models\reward_model_v2.pth"
json_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\rlhf_feedback\preference_pairs_v2.json"
images_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\images\val"
# Load preference pairs
try:
    with open(json_path, "r") as f:
        preference_pairs = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit()

# Use validation set (20% of pairs)
val_size = max(1, int(0.2 * len(preference_pairs)))
val_pairs = preference_pairs[:val_size]

# Initialize models
yolo_model = YOLO(yolo_model_path)
reward_model = RewardModel()
try:
    reward_model.load_state_dict(torch.load(reward_model_path))
    reward_model.eval()
except Exception as e:
    print(f"Error loading reward model: {e}")
    exit()

# Data augmentation
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10)
])

# Function to validate boxes
def validate_box(box, expected_len, img_name, option):
    coords = box[1]
    if not isinstance(coords, list) or len(coords) != expected_len:
        print(f"Invalid box in {img_name} ({option}): {box}, expected {expected_len} coords, got {len(coords)}")
        return False
    if not all(isinstance(c, (int, float, np.floating)) for c in coords):
        print(f"Invalid box values in {img_name} ({option}): {box}")
        return False
    if any(c < 0 for c in coords[:4]):
        print(f"Negative box coordinates in {img_name} ({option}): {box}")
        return False
    return True

# Function to draw boxes
def draw_boxes(image_path, boxes, img_name):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    height, width = img.shape[:2]
    valid_boxes = []
    for box in boxes:
        expected_len = 5 if len(box[1]) > 4 else 4
        if validate_box(box, expected_len, img_name, "option_A" if expected_len == 5 else "option_B"):
            valid_boxes.append(box)
    if not valid_boxes:
        raise ValueError(f"No valid boxes for {img_name}")
    for box in valid_boxes:
        try:
            cls = int(box[0])
            coords = box[1][:4]
            x, y, w, h = map(float, coords)
            x_pixel = x * width
            y_pixel = y * height
            w_pixel = w * width
            h_pixel = h * height
            x1 = max(0, min(int(x_pixel - w_pixel / 2), width - 1))
            y1 = max(0, min(int(y_pixel - h_pixel / 2), height - 1))
            x2 = max(0, min(int(x_pixel + w_pixel / 2), width - 1))
            y2 = max(0, min(int(y_pixel + h_pixel / 2), height - 1))
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            print(f"Drawing box for {img_name}: cls={cls}, top_left={top_left}, bottom_right={bottom_right}")
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error drawing box in {img_name}: {box}, error: {e}")
            continue
    img = cv2.resize(img, (224, 224)) / 255.0
    return img

# Function to compute IoU
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2
    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Evaluate
correct = 0
total = 0
scores_A = []
scores_B = []
ious = []
class_counts = {0: 0, 1: 0, 2: 0}
sample_outputs = []

with torch.no_grad():
    for pair in val_pairs:  # Process all validation pairs
        img_path = os.path.join(images_dir, pair["image"])
        try:
            # Generate YOLO predictions
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            results = yolo_model.predict(img, conf=0.2, imgsz=416)[0]
            yolo_boxes = []
            for box in results.boxes.data.cpu().numpy():
                if len(box) != 6:
                    continue
                x1, y1, x2, y2, conf, cls = box
                x_center = float((x1 + x2) / 2 / img.shape[1])
                y_center = float((y1 + y2) / 2 / img.shape[0])
                width = float((x2 - x1) / img.shape[1])
                height = float((y2 - y1) / img.shape[0])
                box_data = [int(cls), [x_center, y_center, width, height, conf]]
                if validate_box(box_data, 5, pair["image"], "option_A"):
                    yolo_boxes.append(box_data)
                    class_counts[int(cls)] += 1

            # Use ground truth from pair
            gt_boxes = pair["option_B"]
            for box in gt_boxes:
                class_counts[box[0]] += 1

            if not yolo_boxes or not gt_boxes:
                print(f"Skipping {pair['image']}: No valid YOLO predictions ({len(yolo_boxes)}) or ground truth ({len(gt_boxes)})")
                continue

            # Draw and score
            img_A = draw_boxes(img_path, yolo_boxes, pair["image"])
            img_B = draw_boxes(img_path, gt_boxes, pair["image"])
            img_A = torch.tensor(img_A, dtype=torch.float32).permute(2, 0, 1)
            img_B = torch.tensor(img_B, dtype=torch.float32).permute(2, 0, 1)
            img_A = (img_A - 0.5) / 0.5
            img_B = (img_B - 0.5) / 0.5
            img_A = augment(img_A)
            img_B = augment(img_B)
            score_A = reward_model(img_A.unsqueeze(0)).squeeze().item()
            score_B = reward_model(img_B.unsqueeze(0)).squeeze().item()
            scores_A.append(score_A)
            scores_B.append(score_B)

            # Compute average IoU
            avg_iou = 0
            count = 0
            for yolo_box in yolo_boxes:
                for gt_box in gt_boxes:
                    if yolo_box[0] == gt_box[0]:  # Same class
                        iou = compute_iou(yolo_box[1][:4], gt_box[1])
                        avg_iou += iou
                        count += 1
            avg_iou = avg_iou / count if count > 0 else 0
            ious.append(avg_iou)

            # Check if ground truth is ranked higher
            if score_B > score_A:
                correct += 1
            total += 1

            # Save sample output
            if len(sample_outputs) < 3:
                sample_outputs.append({
                    "image": pair["image"],
                    "score_A": score_A,
                    "score_B": score_B,
                    "avg_iou": avg_iou,
                    "yolo_boxes": yolo_boxes,
                    "gt_boxes": gt_boxes
                })

        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping pair {pair['image']}: {e}")
            continue

# Compute metrics
accuracy = correct / total if total > 0 else 0
min_score_A, max_score_A = min(scores_A), max(scores_A)
min_score_B, max_score_B = min(scores_B), max(scores_B)
avg_iou = sum(ious) / len(ious) if ious else 0
total_boxes = sum(class_counts.values())
class_dist = {k: v/total_boxes for k, v in class_counts.items()} if total_boxes > 0 else {0: 0, 1: 0, 2: 0}

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Option A Score Range: [{min_score_A:.4f}, {max_score_A:.4f}]")
print(f"Option B Score Range: [{min_score_B:.4f}, {max_score_B:.4f}]")
print(f"Average IoU: {avg_iou:.4f}")
print(f"Class Distribution: Person={class_dist[0]:.2%}, Weapon={class_dist[1]:.2%}, Violence={class_dist[2]:.2%}")
print(f"Total Pairs Evaluated: {total}")
print(f"Correct Predictions: {correct}")
print("\nSample Outputs:")
for sample in sample_outputs:
    print(f"Image: {sample['image']}")
    print(f"  Score A: {sample['score_A']:.4f}, Score B: {sample['score_B']:.4f}, Avg IoU: {sample['avg_iou']:.4f}")
    print(f"  YOLO Boxes: {sample['yolo_boxes']}")
    print(f"  GT Boxes: {sample['gt_boxes']}")