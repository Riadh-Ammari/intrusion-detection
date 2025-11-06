import os

from ultralytics import YOLO
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OMP fix
import torch
from multiprocessing import freeze_support  # Windows multiprocessing support


# Paths
model_path = r'C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\rlhf\refined_rlhf_20251002_170303\weights\best.pt'
data_yaml = r'C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\data.yaml'

def main():
    # Load original model
    model = YOLO(model_path)
    print("Original model loaded. Simulating FP16 quantization...")

    device = 0 if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Using device: {device}")

    # Step 1: Fuse in FP32 (stable, before half)
    model.fuse()  # Fuse Conv+BN+SiLU in full precision
    print("Model fused in FP32.")

    # Step 2: Convert full model to FP16 (weights + biases)
    model.model.half()  # Half everything
    print("Model converted to FP16.")

    # Step 3: Validate in FP16
    results_fp16 = model.val(data=data_yaml, imgsz=416, batch=8, device=device, half=True)
    baseline_map50 = 0.7800
    print(f"FP16 mAP50: {results_fp16.box.map50:.4f} (delta: {results_fp16.box.map50 - baseline_map50:.4f})")
    print(f"FP16 mAP50-95: {results_fp16.box.map:.4f}")
    print(f"Precision: {results_fp16.box.p.mean():.4f}, Recall: {results_fp16.box.r.mean():.4f}")
    for i, cls in enumerate(['person', 'violence', 'weapon']):
        print(f"{cls} - mAP50: {results_fp16.box.ap50[i]:.4f}, Precision: {results_fp16.box.p[i]:.4f}, Recall: {results_fp16.box.r[i]:.4f}")
    print(f"FP16 Inference Speed: {results_fp16.speed['inference']:.2f} ms/img")

    # Save FP16-ready state_dict
    state_dict = model.model.state_dict()
    torch.save(state_dict, 'best_fp16_state.pt')  # Suppress warning
    print("Saved FP16 state: best_fp16_state.pt (use for ONNX)")

if __name__ == '__main__':
    freeze_support()  # Windows multiprocessing fix
    main()