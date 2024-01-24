"""
    Example code of how the dataset loader can be used.
"""
from torch.utils.data import DataLoader
from .dataset_loader import PletsDataset
import numpy as np
dataset = PletsDataset(".")
dataloader = DataLoader(dataset,batch_size=5)

for i in dataloader:
    print(np.array(i).transpose())
    print(np.array(i).transpose().shape)
    break