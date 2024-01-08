from .base import LanguageModelBase
from transformers import RobertaTokenizer, RobertaModel
from preprocessing.dataset_loader import PletsDataset

class PretrainedRoberta(LanguageModelBase):
    """
        Class for using pretrained RoBERTa models. By default, the 'roberta-base' model is used.
        This class uses the transformers library for loading the pretrained model and tokenizer.
    """
    def __init__(self, pretrained_set_name: str = 'roberta-base'):
        super().__init__()
        self.pretrained_set_name = pretrained_set_name

    def train(self, data: PletsDataset=None) -> 'PretrainedRoberta':
        """
            This class uses a pretrained model and tokenizer, so this function only initializes them.
            Data provided as the parameter of this function is ignored.
        """
        self._model = RobertaModel.from_pretrained(self.pretrained_set_name)
        self._tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_set_name)
        return self