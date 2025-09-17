import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import torchvision.transforms as transforms

# Set environment variable for OpenMP conflict workaround
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RewardModel(nn.Module):
    """CNN model for evaluating bounding box quality."""
    def __init__(self):
        super(RewardModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # For 416x416 input: after 3 pooling operations -> 52x52
        self.fc1 = nn.Linear(64 * 52 * 52, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # ImageNet normalization for preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
    def forward(self, x):
        # Apply normalization
        x = self.normalize(x)
        
        # Convolutional layers with pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def test_reward_model_functionality():
    """Test if RewardModel works and produces different scores for different inputs."""
    print("=== Testing RewardModel Functionality ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize RewardModel
    reward_model = RewardModel().to(device)
    
    # Try to load pre-trained weights
    reward_model_path = "models/reward_model_v4.pth"
    if Path(reward_model_path).exists():
        try:
            checkpoint = torch.load(reward_model_path, map_location=device)
            reward_model.load_state_dict(checkpoint)
            print("✅ RewardModel weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")
            print("Using random initialization")
    else:
        print(f"⚠️ RewardModel weights not found at {reward_model_path}")
        print("Using random initialization")
    
    reward_model.eval()
    
    # Test 1: Different random images should give different scores
    print("\n1. Testing with different random images:")
    
    scores = []
    for i in range(5):
        # Create random image
        test_img = torch.randn(1, 3, 416, 416).to(device)
        
        with torch.no_grad():
            score = reward_model(test_img).item()
            scores.append(score)
            print(f"   Random image {i+1}: Score = {score:.6f}")
    
    score_std = np.std(scores)
    print(f"   Score standard deviation: {score_std:.6f}")
    
    if score_std > 0.01:
        print("✅ RewardModel produces varied scores - working correctly")
    else:
        print("⚠️ RewardModel produces very similar scores - may have issues")
    
    # Test 2: Test with actual images if available
    print("\n2. Testing with rendered bounding boxes:")
    
    # Create a simple test image (416x416)
    base_img = np.ones((416, 416, 3), dtype=np.uint8) * 128  # Gray background
    
    # Image with no boxes
    img_no_boxes = base_img.copy()
    
    # Image with a bounding box
    img_with_box = base_img.copy()
    cv2.rectangle(img_with_box, (100, 100), (300, 300), (255, 0, 0), 2)  # Red box
    cv2.putText(img_with_box, "person", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Convert to tensor and evaluate
    def img_to_tensor(img):
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        return tensor
    
    with torch.no_grad():
        score_no_box = reward_model(img_to_tensor(img_no_boxes)).item()
        score_with_box = reward_model(img_to_tensor(img_with_box)).item()
        
        print(f"   Image with no boxes: Score = {score_no_box:.6f}")
        print(f"   Image with bounding box: Score = {score_with_box:.6f}")
        print(f"   Difference: {abs(score_with_box - score_no_box):.6f}")
        
        if abs(score_with_box - score_no_box) > 0.001:
            print("✅ RewardModel responds to bounding boxes - working correctly")
        else:
            print("⚠️ RewardModel shows little response to bounding boxes")
    
    # Test 3: Reward factor calculation
    print("\n3. Testing reward factor calculation:")
    
    # Simulate advantage calculation
    r_pred = score_with_box
    r_gt = score_no_box
    advantage = r_pred - r_gt
    alpha = 0.1
    
    # Normalize and clamp advantage (simplified)
    normalized_adv = advantage / (abs(advantage) + 1e-8)
    clamped_adv = np.clip(normalized_adv, -2.0, 2.0)
    
    # Compute reward factor
    reward_factor = 1.0 + alpha * (-clamped_adv)
    reward_factor = np.clip(reward_factor, 0.5, 2.0)
    
    print(f"   r_pred: {r_pred:.6f}")
    print(f"   r_gt: {r_gt:.6f}")
    print(f"   advantage: {advantage:.6f}")
    print(f"   reward_factor: {reward_factor:.6f}")
    
    if 0.5 <= reward_factor <= 2.0 and reward_factor != 1.0:
        print("✅ Reward factor calculation working correctly")
    else:
        print("⚠️ Reward factor may not be working as expected")
    
    return True

def check_training_logs():
    """Check if any reward logs exist from previous training runs."""
    print("\n=== Checking Training Logs ===")
    
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("❌ No runs directory found")
        return False
    
    # Look for reward-related directories
    reward_dirs = list(runs_dir.glob("*reward*")) + list(runs_dir.glob("*debug*"))
    
    if not reward_dirs:
        print("❌ No reward training directories found")
        return False
    
    print(f"Found {len(reward_dirs)} reward training directories:")
    
    for directory in reward_dirs:
        print(f"\n📁 {directory.name}:")
        
        # Check for reward logs
        reward_log = directory / "reward_loss_log.txt"
        metrics_log = directory / "reward_metrics.json"
        
        if reward_log.exists():
            size = reward_log.stat().st_size
            if size > 0:
                print(f"   ✅ reward_loss_log.txt ({size} bytes)")
                # Show last few lines
                with open(reward_log, 'r') as f:
                    lines = f.readlines()[-3:]
                    for line in lines:
                        print(f"      {line.strip()}")
            else:
                print(f"   ⚠️ reward_loss_log.txt (empty)")
        else:
            print(f"   ❌ No reward_loss_log.txt")
        
        if metrics_log.exists():
            print(f"   ✅ reward_metrics.json")
        else:
            print(f"   ❌ No reward_metrics.json")
    
    return True

def main():
    """Run all tests to verify reward mechanism."""
    print("🔍 REWARD MECHANISM VERIFICATION")
    print("=" * 50)
    
    # Test RewardModel functionality
    try:
        test_reward_model_functionality()
    except Exception as e:
        print(f"❌ RewardModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check training logs
    try:
        check_training_logs()
    except Exception as e:
        print(f"❌ Log check failed: {e}")
        
    print("\n" + "=" * 50)
    print("🎯 NEXT STEPS:")
    print("1. If RewardModel tests pass, the model itself works")
    print("2. If no reward logs found, the custom trainer isn't being used")
    print("3. Try running: python new_rlhf.py --debug")
    print("4. Look for '[DEBUG]' messages in the output")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    main()