from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

# Paths
best_finetuned_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_detection_v11_reinforce_e50\weights\best_finetuned.pt"
base_model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_detection_v06_e60_20250805_0919\weights\best.pt"
output_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_detection_v11_reinforce_e50\weights\best_finetuned_full.pt"
data_yaml_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\data.yaml"

try:
    # Load the original base model
    model = YOLO(base_model_path)
    # Load the fine-tuned state_dict
    state_dict = torch.load(best_finetuned_path, map_location='cpu', weights_only=True)
    model.model.load_state_dict(state_dict, strict=False)  # Handle mismatches
    model.model.eval()
    print(f"Loaded state_dict from {best_finetuned_path}")
    print(f"Class names: {model.names}")

    # Load data.yaml for metadata
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get('names', ['person', 'violence', 'weapon'])
    nc = data_yaml.get('nc', 3)

    # Load base checkpoint to get its structure
    base_checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
    print(f"Base checkpoint keys: {base_checkpoint.keys()}")

    # Create checkpoint dictionary matching base_checkpoint structure
    checkpoint = {
        'epoch': base_checkpoint.get('epoch', -1),
        'best_fitness': base_checkpoint.get('best_fitness', 0.0),
        'model': model.model.state_dict(),
        'optimizer': base_checkpoint.get('optimizer', None),
        'training_results': base_checkpoint.get('training_results', None),
        'ema': base_checkpoint.get('ema', None),
        'updates': base_checkpoint.get('updates', None),
        'opt': base_checkpoint.get('opt', None),
        'git': base_checkpoint.get('git', None),
        'date': base_checkpoint.get('date', None),
        'nc': nc,
        'names': {i: name for i, name in enumerate(class_names)},
        'yaml': data_yaml,
    }

    # Save checkpoint
    torch.save(checkpoint, output_path, _use_new_zipfile_serialization=False)
    print(f"Saved fixed checkpoint to {output_path}")

    # Verify the checkpoint
    try:
        test_model = YOLO(output_path)
        print(f"Successfully loaded {output_path} for verification")
        print(f"Verification class names: {test_model.names}")
    except Exception as e:
        print(f"Verification failed: {e}")

except Exception as e:
    print(f"Error: {e}")