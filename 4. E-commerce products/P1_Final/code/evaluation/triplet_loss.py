"""
    Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    Modified for the purposes of this project.
"""
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
        Triplet loss function in the form of a pytorch module.
        Takes embeddings of an anchor sample, a positive sample and a negative sample.
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(0)
        distance_negative = (anchor - negative).pow(2).sum(0)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()