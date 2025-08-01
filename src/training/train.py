# === 📚 Importation des bibliothèques ===
import os
import mlflow
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow.pytorch  # Added for model logging

# === ⚙️ Paramètres de configuration ===
project_root = Path(__file__).parent.parent.parent
data_path = project_root / "data" / "dataset_pre" / "data.yaml"
dataset_dvc_path = project_root / "data" / "dataset_pre.dvc"
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
    print(f"Warning: Train images directory not found at {train_dir}")
if not val_dir.exists():
    print(f"Warning: Val images directory not found at {val_dir}")

model_name_base = "intrusion_detection"
model_version = "03"  # Updated to reflect new dataset
full_model_name = f"{model_name_base}_{model_version}"
epochs = 60
batch_size = 16
img_size = 416
project = "runs/train"
name = f"intrusion_detection_v{model_version}_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M')}"

mlflow.set_tracking_uri("file:///" + str(Path.cwd() / "mlruns"))
mlflow.set_experiment("Intrusion_Detection_Trainings")

device = 0 if torch.cuda.is_available() else None
print(f"Using device: {'GPU' if device is not None else 'CPU'}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
            model = YOLO("yolov5nu.pt")  # Using updated model as suggested
            print(f"🎉 Starting training for {full_model_name} with {epochs} epochs on {train_dir} and {val_dir}! 🎉")

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

            # Log metrics as scalars
            mlflow.log_metric("map50-95", results.box.map)  # mAP50-95 is correct
            # Access loss from results.metrics instead of results.box
            if hasattr(results, 'metrics') and 'train/box_loss' in results.metrics:
                mlflow.log_metric("train_box_loss", results.metrics['train/box_loss'][-1])  # Last epoch's loss
            if hasattr(results, 'val') and hasattr(results.val, 'box_loss'):
                mlflow.log_metric("val_box_loss", results.val.box_loss[-1])  # Last epoch's validation loss

            # Plot loss curves
            try:
                train_loss = results.metrics.get("train/box_loss", [0] * epochs) if hasattr(results, 'metrics') else [0] * epochs
                val_loss = results.metrics.get("val/box_loss", [0] * len(train_loss)) if hasattr(results, 'metrics') else [0] * len(train_loss)
                epochs_range = list(range(1, len(train_loss) + 1))

                plt.figure(figsize=(8, 5))
                plt.plot(epochs_range, train_loss, label="Train Loss")
                plt.plot(epochs_range, val_loss, label="Val Loss")
                plt.title("Training vs Validation Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True)

                plot_path = f"{full_model_name}_loss_curve.png"
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots")
                plt.close()
            except Exception as e:
                print("Could not generate loss curves:", e)

            # Log model artifacts
            model_path = f"runs/train/{name}/weights/best.pt"
            if os.path.exists(model_path):
                mlflow.pytorch.log_model(torch.load(model_path, weights_only=True), "model")
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(model_uri=model_uri, name=model_name_base)
                print(f"📦 Model registered as '{model_name_base}', version {registered_model.version}")
            else:
                print("Warning: Model file not found for logging")

            print(f"\n✅ Training completed for {full_model_name}")
            print(f"📦 Run registered for comparison in MLflow")

        except Exception as e:
            print(f"❌ An error occurred: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_text(str(e), "error_details")
            raise