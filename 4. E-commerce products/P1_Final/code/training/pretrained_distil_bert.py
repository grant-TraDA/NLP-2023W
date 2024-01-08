from .base import LanguageModelBase
from transformers import DistilBertTokenizer, DistilBertModel
from preprocessing.dataset_loader import PletsDataset

class PretrainedDistilBert(LanguageModelBase):
    """
        Class for using pretrained DistilBERT models. By default, the 'distilbert-base-uncased' model is used.
        This class uses the transformers library for loading the pretrained model and tokenizer.
    """
    def __init__(self, pretrained_set_name: str = 'distilbert-base-uncased'):
        super().__init__()
        self.pretrained_set_name = pretrained_set_name

    def train(self, data: PletsDataset=None) -> 'PretrainedDistilBert':
        """
            This class uses a pretrained model and tokenizer, so this function only initializes them.
            Data provided as the parameter of this function is ignored.
        """
        self._model = DistilBertModel.from_pretrained(self.pretrained_set_name)
        self._tokenizer = DistilBertTokenizer.from_pretrained(self.pretrained_set_name)
        return self