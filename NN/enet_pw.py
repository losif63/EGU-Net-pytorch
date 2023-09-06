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

class EndmemberNetwork(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
       