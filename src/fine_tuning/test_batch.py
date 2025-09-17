import torch
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import LOGGER
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.cfg import get_cfg, DEFAULT_CFG

# Initialize YOLO model
model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\detect\train33\weights\best.pt"
model = YOLO(model_path, verbose=True)

# Load dataset configuration
data_yaml = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\data.yaml"
cfg = get_cfg(DEFAULT_CFG, {
    'data': data_yaml,
    'batch': 4,
    'imgsz': 416,
    'workers': 8,
    'mode': 'train'
})

# Build dataset and dataloader
dataset = build_yolo_dataset(cfg, data=cfg.data, batch=4, mode='train')
dataloader = build_dataloader(dataset, batch=4, workers=8)

# Get one batch
batch = next(iter(dataloader))

# Move batch to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# Run model inference in predict mode
model.model.eval()
with torch.no_grad():
    preds = model.predict(batch)  # Use predict method to get processed predictions
    LOGGER.info(f"Type of predict output: {type(preds)}")
    if isinstance(preds, list):
        LOGGER.info(f"Predict output is a list with {len(preds)} elements")
        for i, p in enumerate(preds):
            LOGGER.info(f"Element {i} type: {type(p)}, shape: {p.shape if isinstance(p, torch.Tensor) else 'N/A'}")
    else:
        LOGGER.info(f"Predict output shape: {preds.shape if isinstance(preds, torch.Tensor) else 'N/A'}")

# Run model in training mode to inspect raw output
model.model.train()
with torch.no_grad():
    raw_output = model.model(batch)
    LOGGER.info(f"Type of raw model output: {type(raw_output)}")
    if isinstance(raw_output, tuple):
        LOGGER.info(f"Raw model output is a tuple with {len(raw_output)} elements")
        for i, p in enumerate(raw_output):
            LOGGER.info(f"Element {i} type: {type(p)}, shape: {p.shape if isinstance(p, torch.Tensor) else 'N/A'}")
    else:
        LOGGER.info(f"Raw model output shape: {raw_output.shape if isinstance(raw_output, torch.Tensor) else 'N/A'}")

# Test non_max_suppression on predict output
from ultralytics.utils.nms import non_max_suppression
try:
    if isinstance(preds, list):
        nms_preds = non_max_suppression(preds[0].data, conf_thres=0.25, iou_thres=0.45, max_det=300)
    else:
        nms_preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
    LOGGER.info(f"NMS output: {[p.shape for p in nms_preds]}")
except Exception as e:
    LOGGER.error(f"NMS failed: {e}")