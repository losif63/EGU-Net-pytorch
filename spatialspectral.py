import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import scipy.io as scio
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class EndmemberNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc1 = nn.Sequential(
            nn.Conv2d(224, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
            nn.Tanh()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(32, 5, 1, 1, 0),
        )
        self.enc4_softmax = nn.Sequential(
            nn.Softmax(1)
        )

    def forward(self, x, encoder3: nn.Sequential):
        x = self.enc1(x)
        x = self.enc2(x)
        x = encoder3(x)
        x = self.enc4_conv(x)
        self.endmembers = x
        x = self.enc4_softmax(x)
        # self.abundances = x
        return x
    
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
        )
        self.enc3_rest = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2, 2),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.ConvTranspose2d(32, 5, 1, 8, 0, 7),
            nn.Softmax(1)
            # Abundance map generated after here
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
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc3_rest(x)
        x = self.enc4(x)
        self.abundance_map = x
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        # self.recon = x
        return x

class EndmemberGuidedUnmixingNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e_net = EndmemberNetwork()
        self.ur_net = UnmixingReconstructionNetwork()
    
    def forward(self, x_pure, x_mixed):
        self.abundances = self.ur_net.forward(x=x_mixed)
        self.recon = self.e_net.forward(x=x_pure, encoder3=self.ur_net.enc3)
        return self.recon

Pure_TrSet = scio.loadmat('Data/Pure_TrSet.mat')
Mixed_TrSet = scio.loadmat('Data/Mixed_TrSet.mat')
TrLabel = scio.loadmat('Data/TrLabel.mat')
TeLabel = scio.loadmat('Data/TeLabel.mat')

Pure_TrSet = Pure_TrSet['Pure_TrSet']
Mixed_TrSet = Mixed_TrSet['Mixed_TrSet']
TrLabel = TrLabel['TrLabel']
TeLabel = TeLabel['TeLabel']

Pure_TrSet = torch.from_numpy(np.array(Pure_TrSet, dtype=np.float32)).to(device)
Mixed_TrSet = torch.from_numpy(np.array(Mixed_TrSet, dtype=np.float32)).to(device)
TrLabel = torch.from_numpy(np.array(TrLabel, dtype=np.float32)).to(device)
TeLabel = torch.from_numpy(np.array(TeLabel, dtype=np.float32)).to(device)

Y_train = TrLabel
Y_test = TeLabel

x_pure_image = Pure_TrSet.view(-1, 1, 1, 224).permute(0, 3, 1, 2)
x_mixed_image = Mixed_TrSet.view(1, 200, 200, 224).permute(0, 3, 1, 2)

egu_net = EndmemberGuidedUnmixingNetwork()

for epoch in tqdm(range(200)):
    egu_net.forward(x_pure=x_pure_image, x_mixed=x_mixed_image)