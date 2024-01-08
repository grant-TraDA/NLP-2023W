from abc import ABC, abstractmethod
from preprocessing.dataset_loader import PletsDataset


class LanguageModelBase(ABC):
    """
        Abstract class used as a base for all language models used in the pipeline.
    """
    def __init__(self):
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def train(self, data: PletsDataset) -> 'LanguageModelBase':
        """
            Train the model with the given data. Trained models can be accessed through appropriate properties.
            This method needs to be overwritten in each class inheriting from this.
        """
        pass

    @property
    def model(self):
        """
            Proxy for retrieving the trained model.
            Will raise an exception if the model has not been trained yet.
        """
        if self._model is None:
            raise Exception(f"{self.__class__.__name__} must be trained before the model can be accessed.")
        return self._model
    
    @property
    def tokenizer(self):
        """
            Proxy for retrieving the trained tokenizer.
            Will raise an exception if the model has not been trained yet.
        """
        if self._tokenizer is None:
            raise Exception(f"{self.__class__.__name__} must be trained before the tokenizer can be accessed.")
        return self._tokenizer