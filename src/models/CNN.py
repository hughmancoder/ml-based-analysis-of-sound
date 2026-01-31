"""
CNN fromDetecting and Classifying Musical Instruments with
Convolutional Neural Networks

Works with 3 second clip ~300 frames
"""

import torch, torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Input: (B, 2, 128, W) 2-channel mel-spectrogram, variable W (≈ ~300 for 3s @ 10ms hop)
    """
    def __init__(self, in_ch: int = 2, num_classes: int = 0, p_drop: float = 0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, 8,  kernel_size=4, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(8,  16, kernel_size=3, stride=1, padding=0)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        self.bn3   = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2,2)

        self.gap   = nn.AdaptiveAvgPool2d((1,1))   # (B,64,1,1)
        self.drop1 = nn.Dropout(p_drop)     # ~0.5 gives best results in paper
        self.fc1   = nn.Linear(64, 500)
        self.drop2 = nn.Dropout(p_drop)         
        self.fc2   = nn.Linear(500, num_classes)
        

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)   # (B,64,H',W')
        x = self.gap(x)                 # (B,64,1,1)
        x = torch.flatten(x, 1)         # (B,64)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))         # (B,500)
        x = self.drop2(x)
        return self.fc2(x)              
