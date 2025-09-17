import json
import logging
import os
from pathlib import Path
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Configuration ===
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = Path(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_detection_v11_e67_20250822_1025\weights\best.pt")
    TRAIN_DIR = Path(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\images\train")
    LABELS_DIR = Path(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\labels\train")
    OUTPUT_PATH = PROJECT_ROOT / "rlhf_feedback" / "preference_pairs_v2.json"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 416
    CONF_THRESHOLD = 0.3
    MAX_BOXES = 5
    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')
    SIGMOID_K = 10  # Sigmoid steepness for preference mapping

# === IoU Calculation ===
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# === Generate Preference Pairs ===
def generate_preference_pairs():
    logger.info("Starting preference pair generation")
    logger.info(f"Device: {Config.DEVICE}")

    # Verify directories
    if not Config.TRAIN_DIR.exists() or not Config.TRAIN_DIR.is_dir():
        logger.error(f"TRAIN_DIR does not exist or is not a directory: {Config.TRAIN_DIR}")
        raise FileNotFoundError(f"TRAIN_DIR not found: {Config.TRAIN_DIR}")
    if not Config.LABELS_DIR.exists() or not Config.LABELS_DIR.is_dir():
        logger.error(f"LABELS_DIR does not exist or is not a directory: {Config.LABELS_DIR}")
        raise FileNotFoundError(f"LABELS_DIR not found: {Config.LABELS_DIR}")
    if not Config.MODEL_PATH.exists():
        logger.error(f"MODEL_PATH does not exist: {Config.MODEL_PATH}")
        raise FileNotFoundError(f"MODEL_PATH not found: {Config.MODEL_PATH}")

    # Load YOLO model
    try:
        model = YOLO(str(Config.MODEL_PATH))
        model.model.to(Config.DEVICE)
        logger.info(f"YOLO model loaded from {Config.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

    # Collect images
    image_files = [f for f in Config.TRAIN_DIR.glob("*") if f.suffix.lower() in Config.VALID_EXTENSIONS]
    if not image_files:
        logger.error(f"No images found in {Config.TRAIN_DIR} with extensions {Config.VALID_EXTENSIONS}")
        raise FileNotFoundError(f"No images found in {Config.TRAIN_DIR}")
    logger.info(f"Found {len(image_files)} training images")
    logger.debug(f"Sample images: {[f.name for f in image_files[:5]]}")

    # Prioritize images with violence/weapon
    priority_images = []
    regular_images = []
    for img_path in image_files:
        label_path = Config.LABELS_DIR / img_path.with_suffix('.txt').name
        if not label_path.exists():
            logger.warning(f"Label file not found for {img_path.name}")
            continue
        has_weapon_violence = False
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or len(parts) < 5:
                        logger.warning(f"Malformed label in {label_path}: {line.strip()}")
                        continue
                    cls = int(parts[0])
                    if cls in [1, 2]:
                        has_weapon_violence = True
                        break
        except Exception as e:
            logger.warning(f"Error reading label {label_path}: {e}")
            continue
        if has_weapon_violence:
            priority_images.append(img_path)
        else:
            regular_images.append(img_path)

    # Oversample priority images
    image_files = priority_images * 3 + regular_images
    logger.info(f"Processing {len(image_files)} images (oversampled)")

    pairs = []
    class_counts = {0: 0, 1: 0, 2: 0}
    pair_class_counts = {0: 0, 1: 0, 2: 0}
    skipped_images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            skipped_images.append(img_path.name)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(Config.DEVICE)

        # Run YOLO prediction
        try:
            results = model.predict(img_tensor, conf=Config.CONF_THRESHOLD, imgsz=Config.IMG_SIZE)[0]
        except Exception as e:
            logger.warning(f"YOLO prediction failed for {img_path.name}: {e}")
            skipped_images.append(img_path.name)
            continue

        # Extract predicted boxes
        boxes_A = []
        labels_A = []
        pred_boxes = results.boxes.data.cpu().numpy() if results.boxes.data is not None else np.array([])
        selected = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i, 4], reverse=True)[:Config.MAX_BOXES]
        for i in selected:
            x1, y1, x2, y2, conf, cls = pred_boxes[i]
            x_center = float((x1 + x2) / 2 / w)
            y_center = float((y1 + y2) / 2 / h)
            width = float((x2 - x1) / w)
            height = float((y2 - y1) / h)
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                logger.warning(f"Invalid box coordinates in {img_path.name}: {[x_center, y_center, width, height]}")
                continue
            boxes_A.append([x_center, y_center, width, height])
            labels_A.append(int(cls))
            class_counts[int(cls)] += 1

        # Load ground truth boxes
        label_path = Config.LABELS_DIR / img_path.with_suffix('.txt').name
        boxes_B = []
        labels_B = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or len(parts) < 5:
                        logger.warning(f"Malformed ground truth in {img_path.name}: {line.strip()}")
                        continue
                    cls = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        logger.warning(f"Invalid ground truth coordinates in {img_path.name}: {parts[1:5]}")
                        continue
                    boxes_B.append([x_center, y_center, width, height])
                    labels_B.append(cls)
        except Exception as e:
            logger.warning(f"Error reading ground truth {label_path}: {e}")
            skipped_images.append(img_path.name)
            continue

        # Include images with no predictions but valid ground truth
        if not boxes_A and boxes_B:
            pairs.append({
                "image": img_path.name,
                "labels_A": [],
                "boxes_A": [],
                "labels_B": labels_B,
                "boxes_B": boxes_B,
                "preference": 0.1
            })
            pair_class_counts[max(labels_B, default=0)] += 1
            logger.info(f"Generated pair for {img_path.name} (no predictions), preference: 0.1")
            continue
        if not boxes_A or not boxes_B:
            logger.warning(f"Skipping {img_path.name}: No valid predictions ({len(boxes_A)}) or ground truth ({len(boxes_B)})")
            skipped_images.append(img_path.name)
            continue

        # Hungarian matching for IoU
        ious = []
        for cls in set(labels_A + labels_B):
            boxes_A_cls = [b for b, l in zip(boxes_A, labels_A) if l == cls]
            boxes_B_cls = [b for b, l in zip(boxes_B, labels_B) if l == cls]
            if not boxes_A_cls or not boxes_B_cls:
                ious.extend([0.0] * max(len(boxes_A_cls), len(boxes_B_cls)))
                continue
            cost_matrix = np.zeros((len(boxes_A_cls), len(boxes_B_cls)))
            for i, box_A in enumerate(boxes_A_cls):
                for j, box_B in enumerate(boxes_B_cls):
                    cost_matrix[i, j] = 1 - calculate_iou(box_A, box_B)  # Minimize 1-IoU
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                ious.append(1 - cost_matrix[i, j])
            unmatched_A = len(boxes_A_cls) - len(row_ind)
            unmatched_B = len(boxes_B_cls) - len(col_ind)
            ious.extend([0.0] * (unmatched_A + unmatched_B))

        avg_iou = np.mean(ious) if ious else 0.0
        preference = 0.1 + 0.8 * (1 / (1 + np.exp(-Config.SIGMOID_K * (avg_iou - 0.5))))
        preference = max(0.1, min(0.9, preference))

        pairs.append({
            "image": img_path.name,
            "labels_A": labels_A,
            "boxes_A": boxes_A,
            "labels_B": labels_B,
            "boxes_B": boxes_B,
            "preference": round(preference, 2)
        })
        pair_class_counts[max(labels_A + labels_B, default=0)] += 1
        logger.info(f"Generated pair for {img_path.name}, preference: {preference:.2f}")

    # Log statistics
    total_boxes = sum(class_counts.values())
    if total_boxes > 0:
        logger.info(f"Prediction class distribution: Person={class_counts[0]/total_boxes:.2%}, Violence={class_counts[1]/total_boxes:.2%}, Weapon={class_counts[2]/total_boxes:.2%}")
    else:
        logger.warning("No valid boxes found")
    logger.info(f"Pair class distribution: Person={pair_class_counts[0]/len(pairs):.2%}, Violence={pair_class_counts[1]/len(pairs):.2%}, Weapon={pair_class_counts[2]/len(pairs):.2%}")
    if skipped_images:
        logger.warning(f"Skipped {len(skipped_images)} images: {skipped_images[:5]}")
    if len(pairs) < 500:
        logger.warning(f"Only {len(pairs)} pairs generated, consider lowering CONF_THRESHOLD")

    # Save pairs
    try:
        os.makedirs(Config.OUTPUT_PATH.parent, exist_ok=True)
        with open(Config.OUTPUT_PATH, 'w') as f:
            json.dump(pairs, f, indent=2)
        logger.info(f"Saved {len(pairs)} pairs to {Config.OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save preference pairs: {e}")
        raise

# === Main Execution ===
if __name__ == "__main__":
    try:
        generate_preference_pairs()
    except Exception as e:
        logger.error(f"Preference pair generation failed: {e}")
        raise