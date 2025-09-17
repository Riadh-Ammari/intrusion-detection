import cv2
import numpy as np
import torch
import torch.nn as nn
import json
from torchvision import transforms
import random
import os
from fine_tuning.reward_model import RewardModel

# Paths
json_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\rlhf_feedback\preference_pairs_v2.json"
images_dir = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\images\val"
model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\models\reward_model_v2.pth"

# Load preference pairs
try:
    with open(json_path, "r") as f:
        preference_pairs = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit()

if not preference_pairs:
    print("JSON is empty, please rerun generate_preference_pairs.py")
    exit()

# Use validation set (20% of pairs)
random.seed(42)
random.shuffle(preference_pairs)
val_size = max(1, int(0.2 * len(preference_pairs)))
val_pairs = preference_pairs[:val_size]

# Data augmentation (same as training)
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10)
])

# Initialize model
model = RewardModel()
try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

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

# Evaluate model
correct = 0
total = 0
scores_A = []
scores_B = []
class_counts = {0: 0, 1: 0, 2: 0}

with torch.no_grad():
    for pair in val_pairs:
        img_path = os.path.join(images_dir, pair["image"])
        try:
            img_A = draw_boxes(img_path, pair["option_A"], pair["image"])
            img_B = draw_boxes(img_path, pair["option_B"], pair["image"])
            img_A = torch.tensor(img_A, dtype=torch.float32).permute(2, 0, 1)
            img_B = torch.tensor(img_B, dtype=torch.float32).permute(2, 0, 1)
            img_A = (img_A - 0.5) / 0.5
            img_B = (img_B - 0.5) / 0.5
            img_A = augment(img_A)
            img_B = augment(img_B)
            score_A = model(img_A.unsqueeze(0)).squeeze().item()
            score_B = model(img_B.unsqueeze(0)).squeeze().item()
            scores_A.append(score_A)
            scores_B.append(score_B)
            # Count classes
            for box in pair["option_A"]:
                class_counts[box[0]] += 1
            for box in pair["option_B"]:
                class_counts[box[0]] += 1
            # Check if option_B (ground truth) is ranked higher
            if score_B > score_A:
                correct += 1
            total += 1
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping pair {pair['image']}: {e}")
            continue

# Compute metrics
accuracy = correct / total if total > 0 else 0
min_score_A, max_score_A = min(scores_A), max(scores_A)
min_score_B, max_score_B = min(scores_B), max(scores_B)
total_boxes = sum(class_counts.values())
class_dist = {k: v/total_boxes for k, v in class_counts.items()} if total_boxes > 0 else {0: 0, 1: 0, 2: 0}

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Option A Score Range: [{min_score_A:.4f}, {max_score_A:.4f}]")
print(f"Option B Score Range: [{min_score_B:.4f}, {max_score_B:.4f}]")
print(f"Class Distribution: Person={class_dist[0]:.2%}, Weapon={class_dist[1]:.2%}, Violence={class_dist[2]:.2%}")
print(f"Total Pairs Evaluated: {total}")
print(f"Correct Predictions: {correct}")