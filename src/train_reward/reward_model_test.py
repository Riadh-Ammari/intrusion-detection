import cv2
import numpy as np
import torch
import torch.nn as nn

# Preprocessing Function: Draws boxes on the image
# - image_path: Full path to your image (e.g., 416x416 size from YOLO training)
# - boxes: List from preference_pairs.json (e.g., [[class, [x, y, w, h]] or with conf])
# - color_map: Colors for classes (0=violence/red, 1=weapons/green, 2=persons/blue)
def draw_boxes(image_path, boxes, color_map={0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    height, width = img.shape[:2]  # Get actual size (likely 416x416)
    for box in boxes:
        cls = int(box[0])  # Class ID
        coords = box[1]
        x, y, w, h = coords[:4]  # Take first 4 (ignore conf if present)
        # Convert normalized (0-1) to pixels (works for any size, like 416x416)
        x, y, w, h = int(x * width), int(y * height), int(w * width), int(h * height)
        top_left = (x - w // 2, y - h // 2)  # YOLO boxes are center-based
        bottom_right = (x + w // 2, y + h // 2)
        cv2.rectangle(img, top_left, bottom_right, color_map[cls], 2)  # Draw box (thickness=2)
    # Resize to 224x224 for CNN (from 416x416) and normalize (0-1)
    img = cv2.resize(img, (224, 224)) / 255.0
    return img

# Reward Model: Simple CNN Architecture
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential layers: Build the CNN step by step
        self.cnn = nn.Sequential(
            # Layer 1: Conv detects basic features (input 3 channels for RGB)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Output: 16 feature maps
            nn.ReLU(),  # Activation: Adds non-linearity
            nn.MaxPool2d(2),  # Pool: Reduce size to 112x112
            
            # Layer 2: More complex features
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # To 56x56
            
            # Layer 3: Even deeper features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # To 28x28
            
            nn.Flatten(),  # Flatten to 1D vector (64 * 28 * 28 = 50176)
            
            # Dense layers: Connect and reduce to score
            nn.Linear(64 * 28 * 28, 128),  # Fits 224x224 input
            nn.ReLU(),
            nn.Linear(128, 1),  # To single output
            nn.Sigmoid()  # Squeeze to 0-1 score
        )
    
    def forward(self, x):
        return self.cnn(x)  # Pass input through layers

# Test Preprocessing and Model
# Use your new image path and ground truth boxes from the label file
image_path = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\rlhf_feedback\images\2AK-Luger1_jpg.rf.5b87029eca8c44ea4b8de476e548c726.jpg"
# Ground truth boxes from your provided label file
boxes = [
    [2, [0.5612980769230769, 0.5072115384615384, 0.6682692307692307, 0.984375]],  # Person
    [0, [0.6358173076923077, 0.7608173076923077, 0.4831730769230769, 0.47836538461538464]]  # Violence
]
img = draw_boxes(image_path, boxes)
# Save visualized image to check
cv2.imwrite(r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\outputs\tests\test_output.jpg", (img * 255).astype(np.uint8))

# Test the model
model = RewardModel()  # Create the model
img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor (batch, channels, height, width)
score = model(img_tensor)  # Get score
print(f"Test Score (random before training): {score.item():.4f}")  # Should be ~0.5 since untrained