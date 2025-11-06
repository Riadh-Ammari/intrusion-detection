import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OMP fix
import torch
from ultralytics import YOLO
import torch.nn.utils.prune as prune

# Paths
model_path = r'C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\rlhf\refined_rlhf_20251002_170303\weights\best.pt'
data_yaml = r'C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\data.yaml'

# Load model
model = YOLO(model_path)
print("Baseline loaded. Applying 10% permanent pruning to Conv2d layers...")

# Prune (lower 10%, then make permanent)
pruned_count = 0
for module in model.model.model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.1)  # 10% sparsity
        prune.remove(module, 'weight')  # Bake zeros into weights (permanent)
        pruned_count += 1
        print(f"Permanently pruned {module.__class__.__name__} layer")

print(f"Permanently pruned {pruned_count} Conv2d layers.")

# Validate post-prune
results = model.val(data=data_yaml, imgsz=416, batch=8, device='cpu')
baseline_map50 = 0.7800
print(f"Post-prune mAP50: {results.box.map50:.4f} (delta: {results.box.map50 - baseline_map50:.4f})")
print(f"Post-prune mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.p.mean():.4f}, Recall: {results.box.r.mean():.4f}")
for i, cls in enumerate(['person', 'violence', 'weapon']):
    print(f"{cls} - mAP50: {results.box.ap50[i]:.4f}, Precision: {results.box.p[i]:.4f}, Recall: {results.box.r[i]:.4f}")

# Save if good (manual check)
model.save('best_pruned_fixed.pt')
print("Saved: best_pruned_fixed.pt (check metrics; delete if delta <-0.02)")