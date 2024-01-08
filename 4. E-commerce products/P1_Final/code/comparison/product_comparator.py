import torch

from training.base import LanguageModelBase

class ProductComparator:
    """
        Class for producing a multi-hierarchial float similarity metric between two products.
        This class requires a trained language model to be provided, and the input string to be an already attribute-extracted concatenations.
    """
    def __init__(self, model: LanguageModelBase):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.set_model(model)
        self.set_similarity_metric()

    def set_model(self, model: LanguageModelBase) -> 'ProductComparator':
        """
            Set the model and tokenizer to use for embedding calculation.
            The model must be a LanguageModelBase instance.
        """
        self.model = model.model.to(self.device)
        self.tokenizer = model.tokenizer
        return self

    def set_similarity_metric(self, metric: torch.nn.Module = torch.nn.CosineSimilarity(dim=0, eps=1e-6)) -> 'ProductComparator':
        """
            Set the vector similarity metric to use.
            The metric must be a torch module that takes two vectors as input and returns a float value.
            The default metric is CosineSimilarity.
        """
        self.__similarity = metric
        return self

    def embedding(self, text: str) -> torch.Tensor:
        """
            Calculate the embedding of a text with the model.
            This function requires the text to already consist of extracted properties.
        """
        encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model(**encoded_input)
        return output['last_hidden_state'][0,-1,:]

    def similarity_raw(self, text1: str, text2: str) -> torch.Tensor:
        """
            Calculate the similarity of two products based on their embeddings.
            This function requires the text to already consist of extracted properties.
            Returns a tensor float value between 0 and 1, where 0 means no similarity and 1 means identical.
        """
        embed1 = self.embedding(text1)
        embed2 = self.embedding(text2)
        sim = (self.__similarity(embed1, embed2)+1)/2
        return sim
    
    def similarity(self, text1: str, text2: str) -> float:
        """
            Calculate the similarity of two products based on their embeddings.
            This function requires the text to already consist of extracted properties.
            Returns a float value between 0 and 1, where 0 means no similarity and 1 means identical.
        """
        return self.similarity_raw(text1, text2).item()