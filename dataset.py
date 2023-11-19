import os
import scipy.io as scio
from torch.utils.data import Dataset

class HS_Training_Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.labels = scio.loadmat('Data/TrLabel.mat')
        self.labels = self.labels['TrLabel']
        self.spectrums = scio.loadmat('Data/Pure_TrSet.mat')
    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        spectrum = self.spectrums[index]
        label = self.labels[index]
        return spectrum, label

class HS_Test_Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.labels = scio.loadmat('Data/TeLabel.mat')
        self.labels = self.labels['TeLabel']
        self.spectrums = scio.loadmat('Data/Mixed_TrSet.mat')
    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        spectrum = self.spectrums[index]
        label = self.labels[index]
        return spectrum, label
        
