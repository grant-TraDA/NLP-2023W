from .base import LanguageModelBase
from transformers import BertTokenizer, BertModel
from preprocessing.dataset_loader import PletsDataset

class PretrainedBert(LanguageModelBase):
    """
        Class for using pretrained BERT models. By default, the 'bert-base-uncased' model is used.
        This class uses the transformers library for loading the pretrained model and tokenizer.
    """
    def __init__(self, pretrained_set_name: str = 'bert-base-uncased'):
        super().__init__()
        self.pretrained_set_name = pretrained_set_name

    def train(self, data: PletsDataset=None) -> 'PretrainedBert':
        """
            This class uses a pretrained model and tokenizer, so this function only initializes them.
            Data provided as the parameter of this function is ignored.
        """
        self._model = BertModel.from_pretrained(self.pretrained_set_name)
        self._tokenizer = BertTokenizer.from_pretrained(self.pretrained_set_name)
        return self