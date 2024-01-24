"""
    Source: https://github.com/adambielski/siamese-triplet/blob/master/networks.py
    Modified for the purposes of this project.
"""
import torch.nn as nn

class TripletNet(nn.Module):
    """
        A pytorch module serving as a proxy for embedding in the triplet loss function.
        The network takes three inputs, and returns the embeddings of each input.
    """
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)