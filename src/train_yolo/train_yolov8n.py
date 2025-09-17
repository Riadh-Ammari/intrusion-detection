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

def verify_dataset_configuration(data_path):
    """Verify that dataset is properly configured for 3 classes"""
    print("🔍 VERIFYING DATASET CONFIGURATION...")
    
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Data config: {data_config}")
    
    if 'nc' not in data_config or 'names' not in data_config:
        raise ValueError("❌ 'nc' or 'names' not found in data.yaml")
    
    nc = data_config['nc']
    names = data_config['names']
    
    if nc != 3 or len(names) != 3:
        raise ValueError(f"❌ Expected 3 classes, but data.yaml has nc={nc}, names={names}")
    
    print("✅ Dataset configuration verified - 3 classes found")
    return data_config

def test_model_output(model, input_size=(1, 3, 416, 416)):
    """Test model output shape"""
    print("🔍 TESTING MODEL OUTPUT...")
    
    model.model.eval()
    dummy_input = torch.randn(*input_size)
    
    with torch.no_grad():
        try:
            output = model.model(dummy_input)
            
            if isinstance(output, (list, tuple)):
                print(f"Model outputs {len(output)} tensors:")
                for i, out in enumerate(output):
                    print(f"  Output {i}: {out.shape}")
            else:
                print(f"Model output: {output.shape}")
                
            return True
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return False

# === Main Configuration ===
project_root = Path(__file__).parent.parent.parent
data_path = project_root / "data" / "dataset_pre" / "data.yaml"

print(f"Looking for data.yaml at: {data_path.resolve()}")
if not data_path.exists():
    raise FileNotFoundError(f"Data YAML file not found at {data_path}")

# Verify dataset configuration
data_config = verify_dataset_configuration(data_path)

# Training parameters
model_name_base = "intrusion_detection_3class_simple"
model_version = "15"
full_model_name = f"{model_name_base}_{model_version}"
epochs = 100
batch_size = 16
img_size = 416
project = "runs/train"
name = f"intrusion_3class_v{model_version}_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M')}"

mlflow.set_tracking_uri("file:///" + str(Path.cwd() / "mlruns"))
mlflow.set_experiment("Intrusion_Detection_3Class_Simple")

device = 0 if torch.cuda.is_available() else None
print(f"Using device: {'GPU' if device is not None else 'CPU'}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    torch_serialization.add_safe_globals([DetectionModel])

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_name": full_model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "dataset": str(data_path),
            "device": "GPU" if device is not None else "CPU",
            "approach": "let_yolo_handle_architecture",
            "nc": 3,
            "class_names": str(data_config['names'])
        })

        try:
            print("🚀 Initializing YOLOv8n model...")
            
            # Start with fresh YOLOv8n pretrained model
            # YOLO will automatically adapt the head during training initialization
            model = YOLO("yolov8n.pt")
            
            print(f"Initial model nc: {model.nc}")
            print(f"Initial model names: {list(model.names.values())[:10]}...")  # Show first 10 classes
            
            # Test initial model output
            test_model_output(model)
            
            print(f"🎉 Starting training for {full_model_name} with {epochs} epochs!")
            print(f"📊 Training on 3 classes: {data_config['names']}")
            print("🔧 YOLO will automatically adapt the model head for 3 classes during training initialization")

            # Train - let YOLO handle everything automatically
            results = model.train(
                data=str(data_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=project,
                name=name,
                device=device,
                exist_ok=True,
                patience=15,
                workers=2,
                amp=False,
                verbose=True,
                plots=True,
                save_period=10,
                resume=False,  # Don't resume from any previous training
                pretrained=True,  # Use pretrained weights, YOLO will adapt head automatically
                save=True,
                val=True
            )

            print("✅ TRAINING COMPLETED!")
            
            # Verify the trained model
            model_path = f"runs/train/{name}/weights/best.pt"
            if os.path.exists(model_path):
                print("🔍 VERIFYING TRAINED MODEL...")
                
                # Load the trained model
                trained_model = YOLO(model_path)
                print(f"Trained model nc: {trained_model.nc}")
                print(f"Trained model names: {trained_model.names}")
                
                # Test trained model output
                test_model_output(trained_model)
                
                # Verify class count
                if len(trained_model.names) == 3:
                    print("✅ SUCCESS: Model has correct number of classes!")
                else:
                    print(f"⚠️ Warning: Model has {len(trained_model.names)} classes instead of 3")
            
            # Log comprehensive metrics
            if hasattr(results, 'box'):
                metrics = {
                    "map50": float(np.mean(results.box.map50)) if hasattr(results.box, 'map50') else 0.0,
                    "map50-95": float(np.mean(results.box.map)) if hasattr(results.box, 'map') else 0.0,
                    "precision": float(np.mean(results.box.p)) if hasattr(results.box, 'p') else 0.0,
                    "recall": float(np.mean(results.box.r)) if hasattr(results.box, 'r') else 0.0,
                    "classes_count": len(results.names) if hasattr(results, 'names') else 3,
                    "training_completed": True
                }

                # Log class-specific metrics
                class_names = results.names if hasattr(results, 'names') else {0: 'person', 1: 'violence', 2: 'weapon'}
                print(f"Final trained model class names: {class_names}")
                
                if hasattr(results.box, 'ap50'):
                    for i, class_name in class_names.items():
                        if i < len(results.box.ap50):
                            metrics[f"mAP50_{class_name}"] = float(results.box.ap50[i])
                        if hasattr(results.box, 'ap') and i < len(results.box.ap):
                            metrics[f"mAP50-95_{class_name}"] = float(results.box.ap[i])

                mlflow.log_metrics(metrics)
            else:
                print("⚠️ Warning: Could not extract detailed metrics from results")
                mlflow.log_metrics({
                    "training_completed": True,
                    "classes_count": 3
                })

            # Log the model
            if os.path.exists(model_path):
                try:
                    # Log the trained model
                    mlflow.log_artifact(model_path, "model")
                    print(f"📦 Model logged to MLflow")
                    
                except Exception as e:
                    print(f"Warning: Could not log model to MLflow: {e}")

            print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"📊 Model name: {full_model_name}")
            print(f"📦 Classes: {list(class_names.values()) if 'class_names' in locals() else ['person', 'violence', 'weapon']}")
            print(f"🎯 Model saved at: {model_path}")
            
            # Final test with a dummy inference
            if os.path.exists(model_path):
                print("\n🧪 PERFORMING FINAL INFERENCE TEST...")
                final_model = YOLO(model_path)
                
                # Create a dummy image for testing
                dummy_image = torch.randn(416, 416, 3).numpy() * 255
                dummy_image = dummy_image.astype(np.uint8)
                
                try:
                    # Test inference
                    results = final_model(dummy_image, verbose=False)
                    print(f"✅ Inference test successful!")
                    print(f"   Model can process images and return results")
                    if results and len(results) > 0:
                        print(f"   Results shape/type: {type(results[0])}")
                except Exception as e:
                    print(f"⚠️ Inference test failed: {e}")

        except Exception as e:
            print(f"❌ Training failed: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_text(str(e), "error_details")
            raise

    print("\n" + "="*50)
    print("🏁 SCRIPT COMPLETED")
    print("="*50)