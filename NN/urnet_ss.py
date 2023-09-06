import numpy as np
import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class UnmixingReconstructionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(224, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
            nn.AvgPool2d(2, 2),
            nn.Tanh()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2, 2),
            nn.Tanh()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2, 2),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.ConvTranspose2d(32, 5, 1, 8, 0, 7),
            nn.Softmax(1) # softmax on channel dimension N'C'HW
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(5, 32, 1, 1, 0, 0),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 128, 1, 1, 0, 0),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 3, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 224, 5, 1, 2, 0),
            nn.BatchNorm2d(224),
            nn.Tanh()
        )

    def forward(self, x):
        pass
       