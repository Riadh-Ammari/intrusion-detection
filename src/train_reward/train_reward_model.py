# Save as train_reward_model.py
import json
import logging
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path

# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Configuration ===
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PREFERENCE_PATH = PROJECT_ROOT / "rlhf_feedback" / "preference_pairs_v5.json"
    TRAIN_DIR = PROJECT_ROOT / "data" / "dataset_pre" / "images" / "train"
    OUTPUT_PATH = PROJECT_ROOT / "models" / "reward_model_v5.pth"
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    MARGIN = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = (416, 416)
    CLASS_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
    LINE_THICKNESS = 2
    VIOLENCE_WEAPON_WEIGHT = 1.5

# === Reward Model with Edge Channel ===
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (Config.IMG_SIZE[0] // 8) * (Config.IMG_SIZE[1] // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.0],
            std=[0.229, 0.224, 0.225, 1.0]
        )

    def forward(self, x):
        x = self.normalize(x)
        return self.cnn(x)

# === Preference Dataset with Edge Maps ===
class PreferenceDataset(Dataset):
    def __init__(self, preferences, train_dir):
        self.preferences = preferences
        self.train_dir = train_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(Config.IMG_SIZE, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        pref = self.preferences[idx]
        img_path = self.train_dir / pref["image"]
        if not img_path.exists():
            logger.error(f"Image not found: {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge_map = cv2.Canny(gray, 100, 200)
        edge_map = edge_map.astype(np.float32) / 255.0

        img_A = img.copy()
        for label, box in zip(pref["labels_A"], pref["boxes_A"]):
            color = Config.CLASS_COLORS.get(int(label), (255, 255, 255))
            x_center, y_center, width, height = box[:4]
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(img_A, (x1, y1), (x2, y2), color, Config.LINE_THICKNESS)

        img_B = img.copy()
        for label, box in zip(pref["labels_B"], pref["boxes_B"]):
            color = Config.CLASS_COLORS.get(int(label), (255, 255, 255))
            x_center, y_center, width, height = box[:4]
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(img_B, (x1, y1), (x2, y2), color, Config.LINE_THICKNESS)

        img_A_tensor = self.transform(img_A)
        edge_map_resized = cv2.resize(edge_map, Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        edge_tensor = torch.tensor(edge_map_resized, dtype=torch.float32).unsqueeze(0)
        img_A_tensor = torch.cat([img_A_tensor, edge_tensor], dim=0)

        img_B_tensor = self.transform(img_B)
        img_B_tensor = torch.cat([img_B_tensor, edge_tensor], dim=0)

        preference = torch.tensor(pref["preference"], dtype=torch.float32)
        weight = Config.VIOLENCE_WEAPON_WEIGHT if any(int(l) in [1, 2] for l in pref["labels_A"] + pref["labels_B"]) else 1.0

        return img_A_tensor, img_B_tensor, preference, weight

# === Training Function ===
def train_reward_model():
    logger.info("Starting reward model training")
    logger.info(f"Device: {Config.DEVICE}")

    try:
        with open(Config.PREFERENCE_PATH, 'r') as f:
            preferences = json.load(f)
        logger.info(f"Loaded {len(preferences)} preference pairs")
    except Exception as e:
        logger.error(f"Failed to load preferences: {e}")
        raise

    val_size = max(1, int(0.2 * len(preferences)))
    train_pairs = preferences[val_size:]
    val_pairs = preferences[:val_size]
    logger.info(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")
    violence_weapon_pairs = sum(1 for p in preferences if any(int(l) in [1, 2] for l in p["labels_A"] + p["labels_B"]))
    logger.info(f"Violence/Weapon pairs: {violence_weapon_pairs} ({violence_weapon_pairs/len(preferences):.2%})")

    train_dataset = PreferenceDataset(train_pairs, Config.TRAIN_DIR)
    val_dataset = PreferenceDataset(val_pairs, Config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = RewardModel().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MarginRankingLoss(margin=Config.MARGIN)

    best_val_acc = 0.0
    patience = 20
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for img_A, img_B, preference, weight in train_loader:
            img_A, img_B, preference, weight = img_A.to(Config.DEVICE), img_B.to(Config.DEVICE), preference.to(Config.DEVICE), weight.to(Config.DEVICE)
            optimizer.zero_grad()
            score_A = model(img_A).squeeze()
            score_B = model(img_B).squeeze()
            target = 2 * preference - 1
            loss = criterion(score_B, score_A, target)
            weighted_loss = (loss * weight).mean()
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item() * img_A.size(0)
            train_correct += torch.sum((score_B > score_A) == (preference > 0.5)).item()
            train_total += img_A.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img_A, img_B, preference, weight in val_loader:
                img_A, img_B, preference, weight = img_A.to(Config.DEVICE), img_B.to(Config.DEVICE), preference.to(Config.DEVICE), weight.to(Config.DEVICE)
                score_A = model(img_A).squeeze()
                score_B = model(img_B).squeeze()
                target = 2 * preference - 1
                loss = criterion(score_B, score_A, target)
                weighted_loss = (loss * weight).mean()
                val_loss += weighted_loss.item() * img_A.size(0)
                val_correct += torch.sum((score_B > score_A) == (preference > 0.5)).item()
                val_total += img_A.size(0)

        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            try:
                os.makedirs(Config.OUTPUT_PATH.parent, exist_ok=True)
                torch.save(model.state_dict(), str(Config.OUTPUT_PATH))  # Removed weights_only
                logger.info(f"Saved best model to {Config.OUTPUT_PATH}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
                raise
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 4, 416, 416).to(Config.DEVICE)
        outputs = model(dummy_input).squeeze()
        logger.info(f"Test outputs: {outputs.tolist()}, Std: {outputs.std().item():.4f}")

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    try:
        train_reward_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise