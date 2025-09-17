import os
import mlflow
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow.pytorch
import torch.serialization as torch_serialization
from ultralytics.nn.tasks import DetectionModel
import cv2
from typing import List

# Configuration
project_root = Path(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project")
model_path = project_root / "runs" / "train" / "intrusion_detection_v06_e60_20250805_0919" / "weights" / "best.pt"
data_path = project_root / "data" / "dataset_pre" / "data.yaml"
save_dir = project_root / "runs" / "train" / f"intrusion_detection_v10_e30_{datetime.now().strftime('%Y%m%d_%H%M')}"
dataset_dvc_path = project_root / "data" / "dataset_pre.dvc"

# Verify paths
print(f"Looking for model at: {model_path.resolve()}")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")
print(f"Looking for data.yaml at: {data_path.resolve()}")
if not data_path.exists():
    raise FileNotFoundError(f"Data YAML file not found at {data_path}")
with open(data_path, 'r') as f:
    print("Contents of data.yaml:")
    print(f.read())

image_base = data_path.parent / "images"
train_dir = image_base / "train"
val_dir = image_base / "val"
if not train_dir.exists():
    raise FileNotFoundError(f"Train images directory not found at {train_dir}")
if not val_dir.exists():
    raise FileNotFoundError(f"Val images directory not found at {val_dir}")

# Validate dataset
def validate_dataset(image_dir: Path, label_dir: Path) -> List[str]:
    """Check images and labels for validity."""
    errors = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    class_counts = {0: 0, 1: 0, 2: 0}  # person, violence, weapon
    for img_file in image_files:
        img_path = image_dir / img_file
        label_path = label_dir / (img_file.rsplit('.', 1)[0] + '.txt')
        
        # Check image
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"Corrupted image: {img_path}")
            continue
        
        # Check label
        if not label_path.exists():
            errors.append(f"Missing label for: {img_path}")
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                try:
                    values = line.strip().split()
                    if not values:  # Empty line (allowed for empty labels)
                        continue
                    if len(values) != 5:
                        errors.append(f"Malformed label in {label_path}, line {i}: Expected 5 values, got {len(values)}: {line.strip()}")
                        continue
                    cls, x, y, w, h = map(float, values)
                    if not (cls.is_integer() and 0 <= cls <= 2):
                        errors.append(f"Invalid label in {label_path}, line {i}: Invalid class {cls}, must be 0, 1, or 2")
                        continue
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        errors.append(f"Invalid label in {label_path}, line {i}: Coordinates out of range [0, 1]: {line.strip()}")
                    class_counts[int(cls)] += 1
                except:
                    errors.append(f"Malformed label in {label_path}, line {i}: Non-numeric values: {line.strip()}")
    
    return errors, class_counts

# Validate dataset and log stats
print("Validating training dataset...")
train_errors, train_class_counts = validate_dataset(train_dir, data_path.parent / "labels" / "train")
if train_errors:
    print("Training dataset errors:")
    for err in train_errors[:10]:
        print(err)
    if len(train_errors) > 10:
        print(f"... and {len(train_errors) - 10} more errors")
print("Training class counts:", {k: v for k, v in train_class_counts.items()})

print("Validating validation dataset...")
val_errors, val_class_counts = validate_dataset(val_dir, data_path.parent / "labels" / "val")
if val_errors:
    print("Validation dataset errors:")
    for err in val_errors[:10]:
        print(err)
    if len(val_errors) > 10:
        print(f"... and {len(val_errors) - 10} more errors")
print("Validation class counts:", {k: v for k, v in val_class_counts.items()})

# MLflow setup
mlflow.set_tracking_uri("file:///" + str(Path.cwd() / "mlruns"))
mlflow.set_experiment("Intrusion_Detection_Finetune")
run_name = f"intrusion_detection_v10_e30_{datetime.now().strftime('%Y%m%d_%H%M')}"
device = 0 if torch.cuda.is_available() else None
print(f"Using device: {'GPU' if device is not None else 'CPU'}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch_serialization.add_safe_globals([DetectionModel])

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "model_name": "intrusion_detection_v10",
            "base_model": str(model_path),
            "epochs": 30,
            "batch_size": 16,
            "img_size": 416,
            "dataset": str(data_path),
            "dataset_dvc_path": str(dataset_dvc_path),
            "device": "GPU" if device is not None else "CPU",
            "learning_rate": 0.0001,
            "amp": False,
            "augment": True,
            "train_images": len(os.listdir(train_dir)),
            "val_images": len(os.listdir(val_dir)),
            "train_class_counts": train_class_counts,
            "val_class_counts": val_class_counts
        })

        try:
            # Load pre-trained model
            model = YOLO(model_path)
            print(f"🎉 Starting fine-tuning for intrusion_detection_v10 from {model_path} with 30 epochs! 🎉")

            # Fine-tune
            results = model.train(
                data=str(data_path),
                epochs=30,
                batch=16,
                imgsz=416,
                lr0=0.0001,
                optimizer="AdamW",
                project=str(save_dir),
                name="finetune",
                device=device,
                exist_ok=True,
                patience=0,  # Disable early stopping
                workers=2,
                augment=True,
                cos_lr=True,
                amp=False,
                hsv_h=0.01,
                hsv_s=0.2,
                hsv_v=0.2,
                degrees=5.0,
                translate=0.05,
                scale=0.1,
                shear=0.0
            )

            # Log metrics
            metrics = {
                "map50": float(np.mean(results.box.map50)) if results.box.map50 is not None else 0.0,
                "map50-95": float(np.mean(results.box.map)) if results.box.map is not None else 0.0,
                "precision": float(np.mean(results.box.p)) if results.box.p is not None else 0.0,
                "recall": float(np.mean(results.box.r)) if results.box.r is not None else 0.0
            }
            class_names = results.names  # {0: 'person', 1: 'violence', 2: 'weapon'}
            for i, ap50 in enumerate(results.box.ap50):
                metrics[f"mAP50_{class_names[i]}"] = float(ap50)
            for i, ap in enumerate(results.box.ap):
                metrics[f"mAP50-95_{class_names[i]}"] = float(ap)
            for i, p in enumerate(results.box.p):
                metrics[f"precision_{class_names[i]}"] = float(p)
            for i, r in enumerate(results.box.r):
                metrics[f"recall_{class_names[i]}"] = float(r)

            # Log per-epoch loss
            if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
                for key in ["train/box_loss", "val/box_loss"]:
                    if key in results.results_dict:
                        for epoch, val in enumerate(results.results_dict[key], 1):
                            metrics[f"{key.replace('/', '_')}_epoch{epoch}"] = float(val) if not np.isnan(val) else 0.0

            mlflow.log_metrics(metrics)

            # Plot and log loss curves
            try:
                train_box_loss = results.results_dict.get("train/box_loss", [0] * 30)
                val_box_loss = results.results_dict.get("val/box_loss", [0] * 30)
                epochs_range = list(range(1, len(train_box_loss) + 1))

                plt.figure(figsize=(8, 5))
                plt.plot(epochs_range, train_box_loss, label="Train Box Loss", color="#1f77b4")
                plt.plot(epochs_range, val_box_loss, label="Val Box Loss", color="#ff7f0e")
                plt.title("Fine-Tuning: Training vs Validation Box Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True)

                plot_path = save_dir / "finetune" / "loss_curve.png"
                plt.savefig(plot_path)
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
                plt.close()
            except Exception as e:
                print(f"Could not generate loss curves: {e}")

            # Log model
            model_path = save_dir / "weights" / "finetuned.pt"
            if model_path.exists():
                try:
                    mlflow.pytorch.log_model(torch.load(model_path, weights_only=True), "model")
                    model_uri = f"runs:/{run.info.run_id}/model"
                    registered_model = mlflow.register_model(model_uri=model_uri, name="intrusion_detection")
                    print(f"📦 Model registered as 'intrusion_detection', version {registered_model.version}")
                except Exception as e:
                    print(f"Failed to log model to MLflow: {e}")
            else:
                print(f"Warning: Model file not found at {model_path}")

            print(f"\n✅ Fine-tuning completed for intrusion_detection_v10")
            print(f"📦 Model saved to {model_path}")

        except Exception as e:
            print(f"❌ An error occurred: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_text(str(e), "error_details")
            raise