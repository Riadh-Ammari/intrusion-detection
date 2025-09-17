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
from ultralytics.utils.ops import non_max_suppression

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
    OUTPUT_PATH = PROJECT_ROOT / "runs" / "detect" / "fine_tuned_yolo.pt"
    EPOCHS = 20
    BATCH_SIZE = 8  # For GTX 1650 (4GB VRAM)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    ALPHA = 0.1  # Reduced from 0.5 for stability
    IMG_SIZE = (416, 416)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NC = 3
    CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
    LINE_THICKNESS = 2
    REWARD_WARMUP_EPOCHS = 5  # Don't apply reward for first 5 epochs

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
            torch.nn.Linear(128, 1)  # Removed sigmoid for more stable gradients
        )

    def forward(self, x):
        return self.cnn(x)

preprocess_reward = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(Config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def process_predictions(preds, conf_threshold=0.25, iou_threshold=0.6, max_det=50):
    """Process raw YOLO predictions without switching to eval mode."""
    # Apply NMS to raw predictions
    processed = non_max_suppression(
        preds, 
        conf_thres=conf_threshold, 
        iou_thres=iou_threshold, 
        max_det=max_det
    )
    
    batch_boxes = []
    batch_classes = []
    
    for pred in processed:
        if pred is None or len(pred) == 0:
            batch_boxes.append(np.zeros((0, 4), dtype=np.float32))
            batch_classes.append(np.zeros((0,), dtype=np.int64))
        else:
            # Convert from xyxy to xywh normalized
            boxes = pred[:, :4].cpu().numpy()
            # Convert xyxy to xywh
            boxes_xywh = np.zeros_like(boxes)
            boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 / Config.IMG_SIZE[1]  # x_center
            boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / Config.IMG_SIZE[0]  # y_center
            boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / Config.IMG_SIZE[1]      # width
            boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / Config.IMG_SIZE[0]      # height
            
            classes = pred[:, 5].cpu().numpy().astype(int)
            
            batch_boxes.append(boxes_xywh)
            batch_classes.append(classes)
    
    return batch_boxes, batch_classes

def render_boxes(img, boxes, classes):
    """Render boxes on an image for reward model."""
    if torch.is_tensor(img):
        img = img.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Denormalize and clip
    
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
    
    return preprocess_reward(resized).unsqueeze(0).to(Config.DEVICE)

class CustomTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Explicitly set model to ensure it's not a string
        self.model = self.get_model(weights=self.args.model)
        
        # Set hyp explicitly with proper defaults
        self.hyp = SimpleNamespace(
            box=getattr(self.args, 'box', 7.5),
            cls=getattr(self.args, 'cls', 0.5),
            dfl=getattr(self.args, 'dfl', 1.5),
            kobj=getattr(self.args, 'kobj', 1.0),
            label_smoothing=getattr(self.args, 'label_smoothing', 0.0)
        )
        
        # Ensure model.args matches self.hyp
        self.model.args = self.hyp
        
        # Initialize loss criterion
        self.criterion = v8DetectionLoss(model=self.model)
        logger.debug("[CustomTrainer] Loss criterion initialized")
        
        # Initialize reward model
        self.reward_model = RewardModel().to(Config.DEVICE)
        if Config.REWARD_MODEL_PATH.exists():
            self.reward_model.load_state_dict(
                torch.load(str(Config.REWARD_MODEL_PATH), map_location=Config.DEVICE, weights_only=True)
            )
            logger.info(f"Loaded reward model from {Config.REWARD_MODEL_PATH}")
        else:
            logger.warning(f"Reward model not found at {Config.REWARD_MODEL_PATH}, using random weights")
        
        self.reward_model.eval()
        
        # Freeze reward model parameters to prevent updates
        for param in self.reward_model.parameters():
            param.requires_grad = False
            
        self.reward_weight = Config.ALPHA
        self.current_epoch = 0
        
        # Reward tracking
        self.reward_history = []
        
        logger.debug(f"[CustomTrainer] Using hyp: {vars(self.hyp)}")

    def get_model(self, weights=None, cfg=None):
        """Override to use pre-trained model correctly."""
        if isinstance(weights, (YOLO, DetectionModel)):
            model = weights if isinstance(weights, DetectionModel) else weights.model
        elif isinstance(weights, (str, Path)):
            model = YOLO(weights).model
        else:
            raise ValueError(f"Invalid weights type: {type(weights)}. Expected YOLO, DetectionModel, or file path.")
        
        model.to(self.device)
        
        # Ensure all parameters are trainable
        for name, param in model.named_parameters():
            param.requires_grad = True
            
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
        batch = super().preprocess_batch(batch)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == 'img':
                    v = v.to(dtype=torch.float32, non_blocking=True) / 255.0
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def _setup_train(self, world_size):
        """Override to ensure model parameters and buffers are on correct device."""
        super()._setup_train(world_size)
        self.model.to(self.device)
        
        # Ensure reward model is also properly setup
        self.reward_model.to(self.device)
        self.reward_model.eval()

    def _do_train(self, world_size):
        """Override to track epochs for reward warmup."""
        # Track epoch progression for reward warmup
        original_do_train = super()._do_train
        
        def epoch_wrapper():
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                yield from original_do_train(world_size)
                
        return epoch_wrapper()

    def compute_reward_advantage(self, imgs, preds, gt_boxes, gt_classes, batch_idx):
        """Separate method to compute reward advantage with better error handling."""
        batch_size = imgs.shape[0]
        
        # Process predictions directly from forward pass (much faster)
        pred_boxes, pred_classes = process_predictions(preds)
        
        advantages = []
        
        with torch.no_grad():  # Reward computation doesn't need gradients
            for i in range(batch_size):
                try:
                    # Get ground truth for this image
                    mask = batch_idx == i
                    gt_boxes_i = gt_boxes[mask].cpu().numpy() if mask.sum() > 0 else np.zeros((0, 4), dtype=np.float32)
                    gt_classes_i = gt_classes[mask].cpu().numpy().astype(int) if mask.sum() > 0 else np.zeros((0,), dtype=np.int64)

                    # Render images with boxes
                    img = imgs[i]
                    img_pred = render_boxes(img, pred_boxes[i], pred_classes[i])
                    img_gt = render_boxes(img, gt_boxes_i, gt_classes_i)

                    # Compute rewards
                    r_pred = self.reward_model(img_pred).item()
                    r_gt = self.reward_model(img_gt).item()
                    
                    # Compute advantage
                    adv = r_pred - r_gt
                    advantages.append(adv)
                    
                except Exception as e:
                    logger.warning(f"Reward computation failed for image {i}: {e}")
                    advantages.append(0.0)
        
        return torch.tensor(advantages, device=Config.DEVICE, dtype=torch.float32)

    def get_loss(self, batch, preds):
        """Override to add reward weighting with improved stability."""
        # Compute standard YOLO loss
        loss = super().get_loss(batch, preds)
        base_loss = loss.item()
        
        # Apply reward weighting only after warmup period
        if self.current_epoch < Config.REWARD_WARMUP_EPOCHS:
            logger.debug(f"Epoch {self.current_epoch}: Reward warmup, using standard loss: {base_loss}")
            return loss
        
        try:
            # Extract batch components
            imgs = batch['img']
            gt_boxes = batch['bboxes']
            gt_classes = batch['cls'].squeeze(-1)
            batch_idx = batch['batch_idx']
            
            # Compute reward advantages
            advantages = self.compute_reward_advantage(imgs, preds, gt_boxes, gt_classes, batch_idx)
            
            # Normalize advantages for stability
            if advantages.numel() > 1 and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Clamp advantages to prevent extreme scaling
            advantages = torch.clamp(advantages, min=-2.0, max=2.0)
            
            # Compute scaling factor with exponential moving average
            current_factor = torch.mean(advantages)
            if not hasattr(self, 'reward_ema'):
                self.reward_ema = current_factor.item()
            else:
                self.reward_ema = 0.9 * self.reward_ema + 0.1 * current_factor.item()
            
            # Apply reward weighting with gradual increase
            warmup_progress = min(1.0, (self.current_epoch - Config.REWARD_WARMUP_EPOCHS) / 10.0)
            effective_alpha = self.reward_weight * warmup_progress
            
            # Use negative advantage (we want to increase loss when GT is better)
            factor = 1.0 + effective_alpha * (-current_factor)
            factor = torch.clamp(factor, min=0.5, max=2.0)  # Prevent extreme scaling
            
            final_loss = loss * factor
            
            # Track reward metrics
            self.reward_history.append({
                'epoch': self.current_epoch,
                'base_loss': base_loss,
                'reward_factor': factor.item(),
                'final_loss': final_loss.item(),
                'advantages_mean': current_factor.item(),
                'advantages_std': advantages.std().item() if advantages.numel() > 1 else 0.0
            })
            
            logger.debug(f"Epoch {self.current_epoch}: Base loss: {base_loss:.4f}, "
                        f"Reward factor: {factor.item():.4f}, Final loss: {final_loss.item():.4f}")
            
            return final_loss
            
        except Exception as e:
            logger.error(f"Failed to apply reward weighting: {e}")
            return loss  # Fallback to standard loss

    def save_metrics(self):
        """Save reward training metrics for analysis."""
        if hasattr(self, 'reward_history') and self.reward_history:
            import json
            metrics_path = self.save_dir / "reward_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.reward_history, f, indent=2)
            logger.info(f"Saved reward metrics to {metrics_path}")

def main():
    logger.info("Starting reward-weighted fine-tuning with Ultralytics Trainer")
    logger.info(f"Device: {Config.DEVICE}")

    # Load model
    model = YOLO(str(Config.MODEL_PATH))
    logger.info(f"Loaded YOLO model from {Config.MODEL_PATH}, nc={model.model.yaml.get('nc')}")

    # Train with improved stability
    try:
        results = model.train(
            data=str(Config.DATA_PATH),
            epochs=Config.EPOCHS,
            batch=Config.BATCH_SIZE,
            imgsz=Config.IMG_SIZE[0],
            device=Config.DEVICE.index if Config.DEVICE.type == 'cuda' else 'cpu',
            workers=2,
            amp=False,  # Disable AMP for stability with reward model
            lr0=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            name="fine_tuned_yolo_v2",
            trainer=CustomTrainer,
            freeze=None,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            patience=50,  # Increase patience for reward training
            save_period=5   # Save checkpoints more frequently
        )
        logger.info(f"Training complete. Results: {results}")
        
        # Save final metrics
        if hasattr(results, 'trainer'):
            results.trainer.save_metrics()
            
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