import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, x):
        return self.cnn(x)