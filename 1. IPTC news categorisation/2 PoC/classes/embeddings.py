# import tensorflow as tf
# import tensorflow_hub as hub
from transformers import AutoTokenizer, AutoModel, BertTokenizer
import torch
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
from abc import ABC, abstractmethod
import openai


class Embedding(ABC):
    """Abstract class for embeddings."""
    def __init__(self):
        pass

    @abstractmethod
    def get_embedding(self, text):
        """Returns the embedding of the given text."""
        pass


class BertEmbedding(Embedding):
    """Class for BERT embeddings."""
    
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        # if input length is greater than 512 split the input into multiple parts
        
        if len(input_ids[0]) > 512:
            input_ids = input_ids[:, :512]
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]
        return last_hidden_states.squeeze(0).mean(0).detach().numpy()



class Word2VecEmbedding(Embedding):
    """Class for Word2Vec embeddings."""
    
    def __init__(self, word2vec_model=None):
        if word2vec_model:
            self.model = api.load(word2vec_model)
        else:
            self.model = api.load("word2vec-google-news-300")
    
    def get_embedding(self, text):
        words = text.split()
        word_vectors = [self.model[word] for word in words if word in self.model]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)

class OpenaiAdaEmbedding(Embedding):
    """Class for OpenAI Ada embeddings."""

    def __init__(self, api_key):
        openai.api_key = api_key
        self.max_length = 8192

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        response = openai.Embedding.create(
            input=text,
            model=model
        )

        return response['data'][0]['embedding']

# class ELMoEmbedding(Embedding):
    
#     def __init__(self):
#         self.elmo = hub.load("https://tfhub.dev/google/elmo/3")
#         self.embed = self.elmo.signatures["default"]
    
#     def get_embedding(self, text):
#         embeddings = self.embed(tf.constant([text]))
#         return embeddings["elmo"]