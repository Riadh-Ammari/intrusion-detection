import os
import cv2
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils import LOGGER
from torchvision import transforms
import logging
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.loss import v8DetectionLoss

# Set OpenMP workaround for libiomp5md.dll conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train33" / "weights" / "best.pt"
    DATA_PATH = PROJECT_ROOT / "data" / "dataset_pre" / "data.yaml"
    REWARD_MODEL_PATH = PROJECT_ROOT / "models" / "reward_model_v4.pth"
    OUTPUT_PATH = PROJECT_ROOT / "runs" / "fine_tune" / "fine_tuned_yolo.pt"
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    ALPHA = 0.5
    IMG_SIZE = (416, 416)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NC = 3
    CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
    LINE_THICKNESS = 2

# Reward model
class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * (Config.IMG_SIZE[0] // 8) * (Config.IMG_SIZE[1] // 8), 128),
            torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.cnn(x)

preprocess_reward = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(Config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def render_boxes(img, boxes, classes):
    """Render boxes on an image for reward model."""
    if torch.is_tensor(img):
        img = img.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img = (img * 255).astype(np.uint8)  # Denormalize
    resized = cv2.resize(img, Config.IMG_SIZE[::-1])  # (W, H)
    h_res, w_res = resized.shape[:2]
    for box, cls in zip(boxes, classes):
        try:
            x_c, y_c, bw, bh = box
            x1 = int((x_c - bw / 2.0) * w_res)
            y1 = int((y_c - bh / 2.0) * h_res)
            x2 = int((x_c + bw / 2.0) * w_res)
            y2 = int((y_c + bh / 2.0) * h_res)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_res - 1, x2), min(h_res - 1, y2)
            color = Config.CLASS_COLORS.get(int(cls), (255, 255, 255))
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, Config.LINE_THICKNESS)
        except Exception as e:
            logger.warning(f"Failed to render box: {e}")
    return preprocess_reward(resized).to(Config.DEVICE)

class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model, hyp):
        super().__init__(model, hyp)  # Pass hyp as-is, expecting SimpleNamespace
        self.hyp = hyp
        print(self.hyp)
        logger.debug(f"[CustomDetectionLoss] Initialized with hyp: {vars(self.hyp)}")

    def __call__(self, preds, batch):
        """Compute loss using parent class method."""
        logger.debug(f"[CustomDetectionLoss] hyp type: {type(self.hyp)}, content: {vars(self.hyp)}")
        loss, loss_items = super().__call__(preds, batch)
        return loss, loss_items

class CustomTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Normalize hyp early as SimpleNamespace
        print(self.hyp)
        hyp = {
            'box': getattr(self.args, 'box', 7.5),
            'cls': getattr(self.args, 'cls', 0.5),
            'dfl': getattr(self.args, 'dfl', 1.5),
            'kobj': getattr(self.args, 'kobj', 1.0),
            'label_smoothing': getattr(self.args, 'label_smoothing', 0.0),
            'nbs': getattr(self.args, 'nbs', 64)
        }
        self.hyp = SimpleNamespace(**hyp)
        self.args = SimpleNamespace(**vars(self.args))  # Convert args to SimpleNamespace
        self.args.box = self.hyp.box  # Ensure args reflects hyp values
        self.args.cls = self.hyp.cls
        self.args.dfl = self.hyp.dfl
        self.args.kobj = self.hyp.kobj
        self.args.label_smoothing = self.hyp.label_smoothing
        self.args.nbs = self.hyp.nbs
        # Load model
        self.model = self.get_model(weights=self.args.model)
        self.model.args = self.hyp  # Set model.hyp to SimpleNamespace
        logger.debug(f"[CustomTrainer] Initialized hyp: {vars(self.hyp)}")
        logger.debug(f"[CustomTrainer] args: {vars(self.args)}")
        # Override loss function
        self.criterion = CustomDetectionLoss(model=self.model, hyp=self.hyp)
        logger.debug("[CustomTrainer] Loss criterion initialized")
        self.reward_model = RewardModel().to(Config.DEVICE)
        self.reward_model.load_state_dict(torch.load(str(Config.REWARD_MODEL_PATH), map_location=Config.DEVICE, weights_only=True))
        self.reward_model.eval()
        logger.debug(f"Reward model device: {next(self.reward_model.parameters()).device}")
        self.reward_weight = Config.ALPHA
        print(self.hyp)

    def get_model(self, weights=None, cfg=None):
        """Override to use pre-trained model correctly."""
        if isinstance(weights, (YOLO, DetectionModel)):
            model = weights if isinstance(weights, DetectionModel) else weights.model
        elif isinstance(weights, (str, Path)):
            model = YOLO(weights).model  # Load YOLO model from weights file
        else:
            raise ValueError(f"Invalid weights type: {type(weights)}. Expected YOLO, DetectionModel, or file path.")
        model.to(self.device)
        # Check and log parameters and buffers
        for name, param in model.named_parameters():
            param.requires_grad = True
            logger.debug(f"Layer {name} requires_grad={param.requires_grad}, device={param.device}")
        for name, buffer in model.named_buffers():
            logger.debug(f"Buffer {name} device={buffer.device}")
        logger.info(f"Loaded model with nc={model.yaml.get('nc')}, all layers unfrozen")
        return model

    def get_dataloader(self, dataset_path, batch_size, mode='train', workers=2, rank=-1):
        """Implement data loader for training or validation."""
        dataset = build_yolo_dataset(
            cfg=self.args,
            img_path=dataset_path,
            batch=batch_size,
            data=self.data,
            mode=mode,
            rect=False,
            stride=self.model.stride.max().item()
        )
        return build_dataloader(
            dataset=dataset,
            batch=batch_size,
            workers=workers,
            shuffle=(mode == 'train'),
            rank=rank
        )

    def get_validator(self):
        """Implement validator for detection task."""
        return DetectionValidator(
            dataloader=self.get_dataloader(
                dataset_path=self.data.get('val'),
                batch_size=self.args.batch,
                mode='val',
                workers=self.args.workers
            ),
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """Force all batch tensors to the correct device and type before forward pass."""
        batch = super().preprocess_batch(batch)  # Apply Ultralytics augmentations
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == 'img':
                    # Convert to float32 and normalize (0-255 to 0-1)
                    v = v.to(dtype=torch.float32, non_blocking=True) / 255.0
                batch[k] = v.to(self.device, non_blocking=True)
        if "img" in batch:
            logger.debug(f"[preprocess_batch] batch['img'].device = {batch['img'].device}, dtype = {batch['img'].dtype}")
        return batch

    def _setup_train(self, world_size):
        """Override to ensure model parameters and buffers are on correct device."""
        super()._setup_train(world_size)
        # Ensure model and all buffers are on correct device
        self.model.to(self.device)
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                logger.warning(f"Parameter {name} on {param.device}, moving to {self.device}")
                param.data = param.data.to(self.device)
            if name == 'model.22.dfl.conv.weight':
                logger.debug(f"Post-setup: Layer {name} requires_grad={param.requires_grad}, device={param.device}")
        for name, buffer in self.model.named_buffers():
            if buffer.device != self.device:
                logger.warning(f"Buffer {name} on {buffer.device}, moving to {self.device}")
                buffer.data = buffer.to(self.device)
            logger.debug(f"Post-setup: Buffer {name} device={buffer.device}")

    def _do_train(self, world_size):
        """Override to ensure model is on correct device."""
        self.model.to(self.device)  # Ensure model is on correct device
        super()._do_train(world_size)

    def get_loss(self, batch, preds):
        """Override to add reward weighting."""
        logger.debug(f"[get_loss] imgs on {batch['img'].device}, dtype = {batch['img'].dtype}, model on {next(self.model.parameters()).device}")
        loss = super().get_loss(batch, preds)
        logger.debug(f"Standard YOLO loss: {loss.item()}")

        imgs = batch['img']  # [batch_size, 3, H, W]
        gt_boxes = batch['bboxes']  # [n, 4]
        gt_classes = batch['cls'].squeeze(-1)  # [n]
        batch_idx = batch['batch_idx']  # [n]
        batch_size = imgs.shape[0]

        pred_boxes = []
        pred_classes = []
        for i in range(batch_size):
            try:
                res = self.predict(imgs[i:i+1], conf=0.25, iou=0.6, max_det=50)[0]
                if res.boxes is None or len(res.boxes) == 0:
                    pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
                    pred_classes.append(np.zeros((0,), dtype=np.int64))
                else:
                    boxes_xywh = res.boxes.xywh.cpu().numpy()
                    boxes_xywh_norm = boxes_xywh.copy()
                    boxes_xywh_norm[:, 0] /= Config.IMG_SIZE[1]
                    boxes_xywh_norm[:, 1] /= Config.IMG_SIZE[0]
                    boxes_xywh_norm[:, 2] /= Config.IMG_SIZE[1]
                    boxes_xywh_norm[:, 3] /= Config.IMG_SIZE[0]
                    classes_arr = res.boxes.cls.cpu().numpy().astype(int)
                    pred_boxes.append(boxes_xywh_norm)
                    pred_classes.append(classes_arr)
            except Exception as e:
                logger.error(f"Prediction failed for image {i}: {e}")
                pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
                pred_classes.append(np.zeros((0,), dtype=np.int64))

        advs = []
        for i in range(batch_size):
            mask = batch_idx == i
            gt_boxes_i = gt_boxes[mask].cpu().numpy() if mask.sum() > 0 else np.zeros((0, 4), dtype=np.float32)
            gt_classes_i = gt_classes[mask].cpu().numpy().astype(int) if mask.sum() > 0 else np.zeros((0,), dtype=np.int64)

            img = imgs[i]
            img_pred = render_boxes(img, pred_boxes[i], pred_classes[i])
            img_gt = render_boxes(img, gt_boxes_i, gt_classes_i)

            try:
                r_pred = self.reward_model(img_pred).item()
                r_gt = self.reward_model(img_gt).item()
                adv = r_pred - r_gt
                advs.append(adv)
            except Exception as e:
                logger.error(f"Reward computation failed for image {i}: {e}")
                advs.append(0.0)

        try:
            adv_tensor = torch.tensor(advs, device=Config.DEVICE, dtype=torch.float32)
            adv_tensor = torch.clamp(adv_tensor, min=-1.0, max=1.0)
            if adv_tensor.numel() > 1:
                adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
            factor = (1.0 + self.reward_weight * (-adv_tensor)).mean()
            logger.debug(f"Reward factor: {factor.item()}")
            loss = loss * factor
        except Exception as e:
            logger.error(f"Failed to apply reward weighting: {e}")
            factor = 1.0

        return loss

def main():
    logger.info("Starting reward-weighted fine-tuning with Ultralytics Trainer")
    logger.info(f"Device: {Config.DEVICE}")

    # Load model
    model = YOLO(str(Config.MODEL_PATH))
    logger.info(f"Loaded YOLO model from {Config.MODEL_PATH}, nc={model.model.yaml.get('nc')}")

    # Train
    try:
        results = model.train(
            data=str(Config.DATA_PATH),
            epochs=Config.EPOCHS,
            batch=Config.BATCH_SIZE,
            imgsz=Config.IMG_SIZE[0],
            device=Config.DEVICE.index if Config.DEVICE.type == 'cuda' else 'cpu',
            workers=2,
            amp=True,
            lr0=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            name="fine_tuned_yolo",
            trainer=CustomTrainer,
            freeze=None,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64
        )
        logger.info(f"Training complete. Results: {results}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save model
    try:
        model.save(str(Config.OUTPUT_PATH))
        logger.info(f"Model saved to {Config.OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

if __name__ == "__main__":
    main()