import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    """
        A wrapper class for the embedding network in the form of a pytorch module.
        The embedding network is a combination of a base model and a tokenizer.
        The tokenizer is used to tokenize the input, and the base model is used to extract the embeddings.
    """
    def __init__(self, base: nn.Module, tokenizer: nn.Module):
        super(EmbeddingNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.base = base.to(self.device)
        self.tokenizer = tokenizer

    def forward(self, x):
        output = self.tokenizer(x, return_tensors='pt').to(self.device)
        output = self.base(**output)
        return output['last_hidden_state'][0,-1,:]

    def get_embedding(self, x):
        return self.forward(x)