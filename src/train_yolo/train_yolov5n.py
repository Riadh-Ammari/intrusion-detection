import os
import mlflow
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow.pytorch
from ultralytics.nn.tasks import DetectionModel
import torch.serialization as torch_serialization
import yaml
import glob
from torchvision import transforms

# === ⚙️ Paramètres de configuration ===
project_root = Path(__file__).parent.parent.parent
data_path = project_root / "data" / "dataset_pre" / "data.yaml"
dataset_dvc_path = project_root / "data" / "dataset_pre.dvc"
print(f"Looking for data.yaml at: {data_path.resolve()}")
if not data_path.exists():
    raise FileNotFoundError(f"Data YAML file not found at {data_path}")

# Verify data.yaml
with open(data_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
    print("Contents of data.yaml:")
    print(data_yaml)
    if data_yaml.get('nc') != 3 or data_yaml.get('names') != ['person', 'violence', 'weapon']:
        raise ValueError(f"data.yaml must have nc=3 and names=['person', 'violence', 'weapon'], got nc={data_yaml.get('nc')} and names={data_yaml.get('names')}")

image_base = data_path.parent / "images"
train_dir = image_base / "train"
val_dir = image_base / "val"
if not train_dir.exists() or not any(train_dir.glob("*.jpg")):
    raise FileNotFoundError(f"Train images directory not found or empty at {train_dir}")
if not val_dir.exists() or not any(val_dir.glob("*.jpg")):
    raise FileNotFoundError(f"Val images directory not found or empty at {val_dir}")

# Validate labels and count instances
train_labels_dir = data_path.parent / "labels" / "train"
val_labels_dir = data_path.parent / "labels" / "val"
class_counts = {'person': 0, 'violence': 0, 'weapon': 0}
for label_dir, split in [(train_labels_dir, 'train'), (val_labels_dir, 'val')]:
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found at {label_dir}")
    instance_count = 0
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            unique_lines = set(line.strip() for line in lines if line.strip())  # Remove duplicates
            for line in unique_lines:
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Invalid label format in {label_file}: {line}")
                class_id = int(parts[0])
                if class_id > 2:
                    raise ValueError(f"Invalid class ID {class_id} in {label_file}, expected 0, 1, or 2")
                class_name = ['person', 'violence', 'weapon'][class_id]
                class_counts[class_name] += 1
                instance_count += 1
    print(f"{split} instances: {instance_count}")
mlflow.log_metrics({f"{split}_instances_{k}": v for k, v in class_counts.items()})

model_name_base = "intrusion_detection"
model_version = "13"
full_model_name = f"{model_name_base}_{model_version}"
epochs = 4  # Match your test run
batch_size = 16
img_size = 416
project = "runs/train"
name = f"intrusion_detection_v{model_version}_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M')}"

mlflow.set_tracking_uri("file:///" + str(Path.cwd() / "mlruns"))
mlflow.set_experiment("Intrusion_Detection_Trainings")

device = 0 if torch.cuda.is_available() else None
print(f"Using device: {'GPU' if device is not None else 'CPU'}")

# Updated head adaptation function
def adapt_head(model, nc=3):
    """Manually adapt the detection head to nc classes."""
    import torch.nn as nn
    expected_channels = 3 * (nc + 5)  # 3 anchors * (nc classes + 5 box params)
    print("Inspecting model layers for head adaptation...")
    for name, module in model.model.named_modules():
        if name == 'model.24':  # YOLOv5 Detect head
            print(f"Found Detect layer: {name}, type: {type(module)}")
            if hasattr(module, 'cv3'):  # Check for convolutional layer in Detect head
                for i, conv in enumerate(module.cv3):  # cv3 contains the class/bbox prediction conv layers
                    if isinstance(conv, nn.Conv2d):
                        in_channels = conv.in_channels
                        out_channels = conv.out_channels
                        print(f"Layer {name}.cv3[{i}]: in_channels={in_channels}, out_channels={out_channels}")
                        if out_channels != expected_channels:
                            print(f"Adapting head layer {name}.cv3[{i}]: {out_channels} -> {expected_channels}")
                            new_conv = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=expected_channels,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                bias=conv.bias is not None
                            )
                            module.cv3[i] = new_conv
            elif hasattr(module, 'cv2'):  # Alternative for some YOLOv5 variants
                for i, conv in enumerate(module.cv2):
                    if isinstance(conv, nn.Conv2d):
                        in_channels = conv.in_channels
                        out_channels = conv.out_channels
                        print(f"Layer {name}.cv2[{i}]: in_channels={in_channels}, out_channels={out_channels}")
                        if out_channels != expected_channels:
                            print(f"Adapting head layer {name}.cv2[{i}]: {out_channels} -> {expected_channels}")
                            new_conv = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=expected_channels,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                bias=conv.bias is not None
                            )
                            module.cv2[i] = new_conv
            else:
                print(f"Warning: No cv3 or cv2 found in Detect layer {name}. Inspecting all conv layers...")
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.Conv2d):
                        in_channels = sub_module.in_channels
                        out_channels = sub_module.out_channels
                        print(f"Layer {name}.{sub_name}: in_channels={in_channels}, out_channels={out_channels}")
                        if out_channels != expected_channels:
                            print(f"Adapting head layer {name}.{sub_name}: {out_channels} -> {expected_channels}")
                            new_conv = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=expected_channels,
                                kernel_size=sub_module.kernel_size,
                                stride=sub_module.stride,
                                padding=sub_module.padding,
                                bias=sub_module.bias is not None
                            )
                            setattr(module, sub_name, new_conv)
    model.model.yaml['nc'] = nc
    print(f"Updated model.yaml nc to {nc}")
    return model

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Add DetectionModel to safe globals for torch.load
    torch_serialization.add_safe_globals([DetectionModel])

    # Clear any active MLflow runs
    mlflow.end_run()  # Match your working fix

    with mlflow.start_run() as run:
        print(f"Looking for data.yaml at: {data_path.resolve()}")
        with open(data_path, 'r') as f:
            print("Contents of data.yaml:")
            print(f.read())

        mlflow.log_params({
            "model_name": full_model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "dataset": str(data_path),
            "dataset_dvc_path": str(dataset_dvc_path),
            "device": "GPU" if device is not None else "CPU"
        })

        try:
            # Initialize model
            model = YOLO("yolov5nu.pt")
            print(f"Model initialized with yolov5nu.pt, checking nc: {model.model.yaml.get('nc')}")
            mlflow.log_param("initial_nc", model.model.yaml.get('nc'))

            # Adapt head before training
            model = adapt_head(model, nc=3)

            print(f"🎉 Starting training for {full_model_name} with {epochs} epochs on {train_dir} and {val_dir}! 🎉")

            # Train
            results = model.train(
                data=str(data_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=project,
                name=name,
                device=device,
                exist_ok=True,
                patience=10,
                workers=2,
                amp=False
            )

            # Verify model nc and output shape
            print(f"Post-training model nc: {model.model.yaml.get('nc')}")
            print(f"Post-training class names: {model.names}")
            mlflow.log_param("post_training_nc", model.model.yaml.get('nc'))
            mlflow.log_param("post_training_names", str(model.names))

            # Test inference in training mode
            model.model.train()  # Switch to training mode for raw outputs
            test_img = next(val_dir.glob("*.jpg"))
            img = transforms.ToTensor()(plt.imread(str(test_img))[:, :, :3]).unsqueeze(0).to(device)
            if img.max() > 1.0:
                img = img / 255.0  # Normalize
            with torch.no_grad():
                outputs = model.model(img)  # Raw outputs
            output_shapes = [out.shape for out in outputs]
            print(f"Test inference output shapes: {output_shapes}")
            mlflow.log_param("output_shapes", str(output_shapes))

            expected_channels = 3 * (3 + 5)  # 24 for nc=3
            for shape in output_shapes:
                if len(shape) == 4 and shape[1] != expected_channels:
                    print(f"Channel mismatch: expected {expected_channels} channels, got {shape[1]}. Attempting manual head adaptation again.")
                    model = adapt_head(model, nc=3)
                    # Re-test after adaptation
                    with torch.no_grad():
                        outputs = model.model(img)
                    output_shapes = [out.shape for out in outputs]
                    print(f"Output shapes after adaptation: {output_shapes}")
                    mlflow.log_param("output_shapes_after_adaptation", str(output_shapes))
                    for shape in output_shapes:
                        if len(shape) == 4 and shape[1] != expected_channels:
                            raise ValueError(f"Channel mismatch persists: expected {expected_channels} channels, got {shape[1]}")

            # Save adapted model
            model_path = f"runs/train/{name}/weights/best.pt"
            model.save(model_path)

            # Log specified metrics
            metrics = {
                "map50": float(np.mean(results.box.map50)) if results.box.map50 is not None else 0,
                "map50-95": float(np.mean(results.box.map)) if results.box.map is not None else 0,
                "precision": float(np.mean(results.box.p)) if results.box.p is not None else 0,
                "recall": float(np.mean(results.box.r)) if results.box.r is not None else 0,
                "model_params": 2509049,  # From version 07 output
                "train_instances": instance_count,  # From dataset validation
                "val_instances": instance_count,    # From dataset validation
            }

            # Log class-specific metrics
            class_names = results.names
            for i in range(len(class_names)):
                metrics[f"mAP50_{class_names[i]}"] = float(results.box.ap50[i]) if results.box.ap50 is not None else 0
                metrics[f"mAP50-95_{class_names[i]}"] = float(results.box.ap[i]) if results.box.ap is not None else 0
                metrics[f"precision_{class_names[i]}"] = float(results.box.p[i]) if results.box.p is not None else 0
                metrics[f"recall_{class_names[i]}"] = float(results.box.r[i]) if results.box.r is not None else 0

            # Log per-epoch loss metrics
            if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
                for key in ["train/box_loss", "val/box_loss"]:
                    if key in results.results_dict:
                        for epoch, val in enumerate(results.results_dict[key], 1):
                            metrics[f"{key.replace('/', '_')}_epoch{epoch}"] = float(val)

            # Log all metrics to MLflow
            mlflow.log_metrics(metrics)

            # Plot and log loss curves
            try:
                train_box_loss = results.results_dict.get("train/box_loss", [0] * epochs)
                val_box_loss = results.results_dict.get("val/box_loss", [0] * epochs)
                epochs_range = list(range(1, len(train_box_loss) + 1))

                plt.figure(figsize=(8, 5))
                plt.plot(epochs_range, train_box_loss, label="Train Box Loss", color="#1f77b4")
                plt.plot(epochs_range, val_box_loss, label="Val Box Loss", color="#ff7f0e")
                plt.title("Training vs Validation Box Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True)

                plot_path = f"{full_model_name}_loss_curve.png"
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots")
                plt.close()
            except Exception as e:
                print(f"Could not generate loss curves: {e}")

            # Log model artifacts
            if os.path.exists(model_path):
                try:
                    mlflow.pytorch.log_model(torch.load(model_path, weights_only=True), "model")
                    model_uri = f"runs:/{run.info.run_id}/model"
                    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name_base)
                    print(f"📦 Model registered as '{model_name_base}', version {registered_model.version}")
                except Exception as e:
                    print(f"Failed to log model to MLflow: {e}")
            else:
                print(f"Warning: Model file not found at {model_path}")

            print(f"\n✅ Training completed for {full_model_name}")
            print(f"📦 Run registered for comparison in MLflow")

        except Exception as e:
            print(f"❌ An error occurred: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_text(str(e), "error_details")
            raise