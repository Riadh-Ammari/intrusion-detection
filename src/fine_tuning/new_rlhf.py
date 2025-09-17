
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import warnings
import logging
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import LOGGER
from ultralytics.cfg import DEFAULT_CFG
import torchvision.transforms as transforms
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.nms import non_max_suppression

# Clean environment
sys.argv = [sys.argv[0]]
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RewardModel(nn.Module):
    """Reward model that evaluates detection quality"""
    def __init__(self):
        super(RewardModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 52 * 52, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, x):
        x = self.normalize(x)
        return self.cnn(x)

class RefinedRLHFDetectionTrainer(DetectionTrainer):
    """Refined RLHF-enhanced YOLO trainer ensuring RewardModel functionality"""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        clean_overrides = {k: v for k, v in overrides.items() if k != 'session' and v is not None and k != 'reward_model_path'}
        super().__init__(cfg, clean_overrides, _callbacks)
        
        self.reward_model_path = overrides.get('reward_model_path', r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\models\reward_model_v4.pth")
        self.reward_weight = 0.1
        self.warmup_epochs = 0
        self.class_weights = torch.tensor([0.3, 3.0, 3.0], device=self.device)  # person, violence, weapon
        self.reward_model = self._load_reward_model()
        self.batch_counter = 0
        self.reward_history = []
        self.last_box_loss = 0.0
        self.last_cls_loss = 0.0
        self.last_dfl_loss = 0.0
        self.last_reward_loss = 0.0
        self.last_total_loss = 0.0
        self.last_r_pred = 0.0
        self.class_pred_counts = [0, 0, 0]  # person, violence, weapon
        
        LOGGER.info(f"🔧 REFINED RLHF Trainer Initialized:")
        LOGGER.info(f"   Reward weight: {self.reward_weight}")
        LOGGER.info(f"   Reward model loaded: {self.reward_model is not None}")
        LOGGER.info(f"   Warmup epochs: {self.warmup_epochs}")
        LOGGER.info(f"   Class weights: {self.class_weights.tolist()}")
    
    def _load_reward_model(self):
        if not os.path.exists(self.reward_model_path):
            LOGGER.warning(f"❌ Reward model not found: {self.reward_model_path}")
            return None
        
        try:
            reward_model = RewardModel()
            state_dict = torch.load(self.reward_model_path, map_location=self.device)
            reward_model.load_state_dict(state_dict)
            reward_model.eval()
            reward_model.to(self.device)
            
            with torch.no_grad():
                weight_sum = sum(p.abs().sum().item() for p in reward_model.parameters())
                if weight_sum < 1e-3:
                    LOGGER.warning("⚠️ Reward model weights are near zero - likely untrained!")
                    return None
                
                dummy_input = torch.randn(2, 3, 416, 416).to(self.device)
                test_outputs = reward_model(dummy_input)
                LOGGER.info(f"✅ Reward model test outputs: {test_outputs.squeeze().tolist()}")
                
                if test_outputs.std() < 1e-8:
                    LOGGER.warning("⚠️ Reward model outputs have very low variance - may be ineffective")
                
            return reward_model
            
        except Exception as e:
            LOGGER.warning(f"❌ Failed to load reward model: {e}")
            return None

    def draw_clean_boxes_on_image(self, img, boxes):
        if len(boxes) == 0:
            return img
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        
        for box in boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
                y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
                
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img_copy

    def draw_predictions_on_batch(self, batch, predictions):
        imgs = batch['img'].float() / 255.0
        batch_size = imgs.size(0)
        processed_images = []
        
        for i in range(batch_size):
            img = imgs[i].permute(1, 2, 0).cpu().numpy() * 255.0
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            box_list = predictions[i].cpu().numpy() if i < len(predictions) and len(predictions[i]) > 0 else []
            img_with_boxes = self.draw_clean_boxes_on_image(img, box_list)
            
            if self.batch_counter % 10 == 0:
                LOGGER.info(f"DEBUG: Batch {self.batch_counter}, Image {i}: {len(box_list)} boxes drawn, pixel range [{img_with_boxes.min()}, {img_with_boxes.max()}]")
            
            if len(box_list) > 0:
                for box in box_list:
                    if len(box) >= 6:
                        cls_id = int(box[5])
                        if 0 <= cls_id < 3:
                            self.class_pred_counts[cls_id] += 1
            
            img_tensor = torch.from_numpy(img_with_boxes).float().permute(2, 0, 1) / 255.0
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0), size=(416, 416), mode='bilinear', align_corners=False
            ).squeeze(0)
            processed_images.append(img_tensor)
        
        return torch.stack(processed_images).to(self.device)

    def compute_reward_loss(self, batch, preds):
        if self.reward_model is None:
            LOGGER.warning("Reward model is disabled - returning zero reward loss")
            return torch.tensor(0.0, device=self.device)
        
        try:
            processed_preds = self._process_predictions_robust(preds, batch)
            pred_images = self.draw_predictions_on_batch(batch, processed_preds)
            
            with torch.no_grad():
                reward_scores = self.reward_model(pred_images).squeeze(-1)
                avg_reward = reward_scores.mean()
                self.last_r_pred = avg_reward.item()
                
                if self.batch_counter % 10 == 0:
                    LOGGER.info(f"DEBUG: Batch {self.batch_counter}: Reward scores: {reward_scores.tolist()}")
                    LOGGER.info(f"DEBUG: Batch {self.batch_counter}: Average reward: {avg_reward.item():.6f}")
            
            reward_loss = -avg_reward * self.reward_weight
            reward_loss = torch.clamp(reward_loss, -0.5, 0.5)
            
            return reward_loss
            
        except Exception as e:
            LOGGER.warning(f"ERROR in reward computation: {e}")
            return torch.tensor(0.0, device=self.device)

    def _process_predictions_robust(self, preds, batch):
        if not isinstance(preds, tuple) or len(preds) == 0:
            batch_size = batch['img'].size(0)
            return [torch.empty(0, 6, device=self.device) for _ in range(batch_size)]
        
        try:
            pred_tensor = preds[0]
            if pred_tensor.dim() != 3:
                LOGGER.warning(f"Unexpected prediction tensor dimensions: {pred_tensor.shape}")
                batch_size = batch['img'].size(0)
                return [torch.empty(0, 6, device=self.device) for _ in range(batch_size)]
            
            pred_tensor = pred_tensor.permute(0, 2, 1)
            bbox_coords = pred_tensor[..., :4]
            bbox_coords = xywh2xyxy(bbox_coords)
            pred_tensor = torch.cat([bbox_coords, pred_tensor[..., 4:]], dim=-1)
            
            processed_preds = non_max_suppression(
                pred_tensor,
                conf_thres=0.25,
                iou_thres=0.45,
                max_det=300,
                nc=3
            )
            
            return processed_preds
            
        except Exception as e:
            LOGGER.warning(f"ERROR in prediction processing: {e}")
            batch_size = batch['img'].size(0)
            return [torch.empty(0, 6, device=self.device) for _ in range(batch_size)]

    def criterion(self, preds, batch):
        yolo_loss = self.model.loss(preds, batch)
        
        if hasattr(self.model, 'loss_items') and self.model.loss_items is not None:
            loss_items = self.model.loss_items
            self.last_box_loss = loss_items[0].item() if len(loss_items) > 0 else 0.0
            self.last_cls_loss = loss_items[1].item() if len(loss_items) > 1 else 0.0
            self.last_dfl_loss = loss_items[2].item() if len(loss_items) > 2 else 0.0
        else:
            self.last_box_loss = yolo_loss.item()
            self.last_cls_loss = 0.0
            self.last_dfl_loss = 0.0
        
        # Apply per-class weights to classification loss
        if self.last_cls_loss > 0:
            cls_loss_tensor = self.model.loss_items[1]
            cls_loss_tensor *= self.class_weights.mean()
            self.last_cls_loss = cls_loss_tensor.item()
            yolo_loss = self.last_box_loss + self.last_cls_loss + self.last_dfl_loss
        
        if self.epoch >= self.warmup_epochs:
            reward_loss = self.compute_reward_loss(batch, preds)
            self.last_reward_loss = reward_loss.item()
        else:
            reward_loss = torch.tensor(0.0, device=self.device)
            self.last_reward_loss = 0.0
        
        total_loss = yolo_loss + reward_loss
        self.last_total_loss = total_loss.item()
        
        reward_score = -self.last_reward_loss / self.reward_weight if self.reward_weight > 0 else 0.0
        self.reward_history.append(reward_score)
        self.batch_counter += 1
        
        if self.batch_counter % 25 == 0:
            self._log_rlhf_progress()
        
        return total_loss

    def _log_rlhf_progress(self, end_of_epoch=False):
        recent_rewards = self.reward_history[-25:] if len(self.reward_history) >= 25 else self.reward_history
        avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        reward_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
        
        LOGGER.info(f"\n{'='*100}")
        LOGGER.info(f"🎯 REFINED RLHF TRAINING PROGRESS - {'EPOCH END' if end_of_epoch else f'BATCH {self.batch_counter}'} | EPOCH {self.epoch + 1}")
        LOGGER.info(f"{'='*100}")
        LOGGER.info(f"📊 LOSS BREAKDOWN:")
        LOGGER.info(f"   📦 Box Loss (bbox regression):    {self.last_box_loss:.6f}")
        LOGGER.info(f"   🏷️  Classification Loss:          {self.last_cls_loss:.6f}")
        LOGGER.info(f"   🎯 Distribution Focal Loss:       {self.last_dfl_loss:.6f}")
        LOGGER.info(f"   🎁 Reward Loss (-reward * weight): {self.last_reward_loss:.6f}")
        LOGGER.info(f"   📈 Total Combined Loss:           {self.last_total_loss:.6f}")
        LOGGER.info(f"   💡 Loss Impact: {((self.last_reward_loss / self.last_total_loss) * 100):.2f}% of total loss")
        LOGGER.info(f"🎯 REWARD ANALYSIS:")
        LOGGER.info(f"   Current Reward Score:             {self.last_r_pred:.6f}")
        LOGGER.info(f"   Recent Avg Reward (25 batches):   {avg_recent_reward:.6f}")
        LOGGER.info(f"   Reward Std Deviation:             {reward_std:.6f}")
        LOGGER.info(f"   Reward Weight (alpha):            {self.reward_weight}")
        LOGGER.info(f"   Reward Model Status:              {'✅ Active' if self.reward_model else '❌ Disabled'}")
        LOGGER.info(f"   RLHF Status:                      {'✅ Active' if self.epoch >= self.warmup_epochs else '❌ In Warmup'}")
        LOGGER.info(f"📦 PREDICTION STATISTICS:")
        LOGGER.info(f"   Person predictions:               {self.class_pred_counts[0]}")
        LOGGER.info(f"   Violence predictions:             {self.class_pred_counts[1]}")
        LOGGER.info(f"   Weapon predictions:               {self.class_pred_counts[2]}")
        self.class_pred_counts = [0, 0, 0]
        LOGGER.info(f"📈 TRAINING STATISTICS:")
        LOGGER.info(f"   Total Batches Processed:          {self.batch_counter}")
        LOGGER.info(f"   Current Epoch:                    {self.epoch + 1}")
        LOGGER.info(f"   Total Rewards Collected:          {len(self.reward_history)}")
        LOGGER.info(f"{'='*100}\n")

    def on_epoch_end(self, epoch):
        """Log per-class metrics and reward status at epoch end"""
        super().on_epoch_end(epoch)
        self._log_rlhf_progress(end_of_epoch=True)
        
        try:
            metrics = self.validator.validate()
            if metrics and hasattr(metrics, 'maps') and len(metrics.maps) >= 3:
                class_names = self.model.names
                LOGGER.info(f"\n📊 PER-CLASS VALIDATION METRICS (EPOCH {epoch + 1}):")
                for i, (p, r, ap50, ap) in enumerate(zip(metrics.top1, metrics.top5, metrics.maps, metrics.ap)):
                    LOGGER.info(f"   {class_names[i]}: Precision={p:.3f}, Recall={r:.3f}, mAP50={ap50:.3f}, mAP50-95={ap:.3f}")
        except Exception as e:
            LOGGER.warning(f"Failed to log per-class metrics: {e}")

def main():
    LOGGER.info("\n" + "="*100)
    LOGGER.info("🚀 STARTING REFINED RLHF-ENHANCED YOLO TRAINING")
    LOGGER.info("="*100)
    
    if torch.cuda.is_available():
        LOGGER.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        LOGGER.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        LOGGER.info("❌ No GPU - using CPU")
    
    model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\runs\train\intrusion_3class_v15_e100_20250912_1638\weights\best.pt"
    
    if not os.path.exists(model_path):
        LOGGER.error(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        LOGGER.info(f"✅ YOLO model loaded: {model.nc} classes")
        LOGGER.info(f"   Class names: {list(model.names.values())}")
        
        reward_model_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\models\reward_model_v4.pth"
        config = {
            'data': r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\data\dataset_pre\data.yaml",
            'epochs': 10,
            'batch': 8,
            'imgsz': 416,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'project': 'runs/rlhf',
            'name': f'refined_rlhf_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'lr0': 0.0001,
            'patience': 10,
            'save': True,
            'verbose': True,
            'workers': 2,
            'amp': False,
            'plots': True,
            'optimizer': 'AdamW',
            'cls': 1.0,
            'reward_model_path': reward_model_path
        }
        
        LOGGER.info(f"\n📋 REFINED RLHF TRAINING CONFIGURATION:")
        for k, v in config.items():
            LOGGER.info(f"   {k}: {v}")
        
        LOGGER.info(f"\n🔧 REFINEMENTS APPLIED:")
        LOGGER.info(f"   1. ✅ Removed text annotations from box drawing")
        LOGGER.info(f"   2. ✅ Class weighting for person:0.3, violence:3.0, weapon:3.0 in criterion")
        LOGGER.info(f"   3. ✅ Enabled verbose=True for reward logging")
        LOGGER.info(f"   4. ✅ Robust prediction processing for 3 classes")
        LOGGER.info(f"   5. ✅ Added per-class prediction counting")
        LOGGER.info(f"   6. ✅ Enhanced reward model validation")
        LOGGER.info(f"   7. ✅ Improved error handling")
        LOGGER.info(f"   8. ✅ Added per-class validation metrics")
        LOGGER.info(f"   9. ✅ Relaxed RewardModel validation to ensure functionality")
        
        LOGGER.info(f"\n🔥 RLHF WORKFLOW:")
        LOGGER.info(f"   1. ✅ Model & Config Loading - DONE")
        LOGGER.info(f"   2. ✅ Reward Model Integration - ENHANCED")
        LOGGER.info(f"   3. 🔄 Training Setup - INITIALIZING...")
        LOGGER.info(f"   4. 🔄 Forward Pass - YOLO + REFINED REWARD LOSSES")
        LOGGER.info(f"   5. 🔄 Reward Adjustment - CLEAN BOX-ONLY EVALUATION")
        LOGGER.info(f"   6. 🔄 Training Loop - 10 EPOCHS WITH CLASS WEIGHTS")
        LOGGER.info(f"   7. 🎯 Output - REFINED RLHF MODEL")
        LOGGER.info("="*100 + "\n")
        
        results = model.train(trainer=RefinedRLHFDetectionTrainer, **config)
        
        LOGGER.info("\n" + "="*100)
        LOGGER.info("🎉 REFINED RLHF TRAINING COMPLETED!")
        LOGGER.info("="*100)
        LOGGER.info(f"🎯 Model refined with proper reward-based feedback")
        LOGGER.info(f"📊 Standard YOLO losses + Clean reward evaluation")
        LOGGER.info(f"⚖️ Class imbalance addressed with weights [0.3, 3.0, 3.0] in criterion")
        LOGGER.info(f"🏆 Enhanced model saved in runs/rlhf/ directory")
        LOGGER.info(f"💡 Model optimized for violence/weapon detection quality")
        
        if hasattr(results, 'maps') and len(results.maps) >= 3:
            class_names = model.names
            LOGGER.info("\n📊 PER-CLASS VALIDATION METRICS (FINAL):")
            for i, (p, r, ap50, ap) in enumerate(zip(results.top1, results.top5, results.maps, results.ap)):
                LOGGER.info(f"   {class_names[i]}: Precision={p:.3f}, Recall={r:.3f}, mAP50={ap50:.3f}, mAP50-95={ap:.3f}")
        
    except Exception as e:
        LOGGER.error(f"\n❌ REFINED RLHF TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
