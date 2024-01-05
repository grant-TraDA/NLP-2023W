import os
from typing import Tuple

import numpy as np
import pandas as pd
from keras import Model
from keras.layers import Input, Embedding, Dense, GlobalMaxPooling1D, Conv1D, TextVectorization, \
    Dropout, Concatenate, Bidirectional, GRU
from keras.optimizers import Adam


def prepare_data(directory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data
    :param directory: Directory with data
    :return: Dataframes with data
    """
    train_df = pd.read_csv(os.path.join(directory, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(directory, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(directory, 'test.csv'))

    def get_data(df):
        x = df['text'].values
        y = np.where(df["account.type"] == "bot", 1, 0)
        return x, y

    x_train, y_train = get_data(train_df)
    x_valid, y_valid = get_data(valid_df)
    x_test, y_test = get_data(test_df)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def prepare_text_vectorizer(x_train: np.array, max_len: int, vocab_size: int) -> TextVectorization:
    """
    Prepare text vectorizer
    :param x_train: Training data
    :param max_len: Max length of sentence
    :param vocab_size: Size of vocabulary
    :return: Text vectorizer
    """
    text_vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
    text_vectorizer.adapt(x_train)
    return text_vectorizer


def get_wordcnn(text_vectorizer: TextVectorization) -> Model:
    """
    Get WordCNN model
    :param text_vectorizer: Text vectorizer adapted to a training data
    :return: Keras model
    """
    input_layer = Input(shape=(1,), dtype="string")
    vocab_size = len(text_vectorizer.get_vocabulary())
    x = text_vectorizer(input_layer)
    x = Embedding(vocab_size, 128)(x)
    num_filters = 128
    x3 = Conv1D(num_filters, 3, activation="tanh")(x)
    x3 = GlobalMaxPooling1D()(x3)

    x4 = Conv1D(num_filters, 4, activation="tanh")(x)
    x4 = GlobalMaxPooling1D()(x4)

    x5 = Conv1D(num_filters, 5, activation="tanh")(x)
    x5 = GlobalMaxPooling1D()(x5)

    conc = Concatenate()([x3, x4, x5])
    conc = Dropout(0.2)(conc)

    final_out = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=input_layer, outputs=final_out)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss=['binary_crossentropy'], optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def get_wordgru(text_vectorizer: TextVectorization) -> Model:
    """
    Get WordGRU model
    :param text_vectorizer: Text vectorizer adapted to a training data
    :return: Keras model
    """
    vocab_size = len(text_vectorizer.get_vocabulary())
    input_layer = Input(shape=(1,), dtype="string")
    x = text_vectorizer(input_layer)
    x = Embedding(vocab_size, 128)(x)
    num_filters = 512

    gru = Bidirectional(GRU(num_filters, activation="tanh"))(x)
    conc = Dropout(0.2)(gru)
    final_out = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=input_layer, outputs=final_out)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss=['binary_crossentropy'], optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def get_wordcnn_gru(text_vectorizer: TextVectorization) -> Model:
    """
    Get WordCNN-GRU model
    :param text_vectorizer: Text vectorizer adapted to a training data
    :return: Keras model
    """
    vocab_size = len(text_vectorizer.get_vocabulary())
    input_layer = Input(shape=(1,), dtype="string")
    x = text_vectorizer(input_layer)
    x = Embedding(vocab_size, 128)(x)
    num_cnn_filters = 128
    num_gru_filters = 512
    x3 = Conv1D(num_cnn_filters, 3, activation="tanh")(x)
    x3 = GlobalMaxPooling1D()(x3)

    x4 = Conv1D(num_cnn_filters, 4, activation="tanh")(x)
    x4 = GlobalMaxPooling1D()(x4)

    x5 = Conv1D(num_cnn_filters, 5, activation="tanh")(x)
    x5 = GlobalMaxPooling1D()(x5)

    conc = Concatenate()([x3, x4, x5])
    conc = Dropout(0.2)(conc)

    gru = Bidirectional(GRU(num_gru_filters, activation="tanh"))(conc)
    x = Dropout(0.2)(gru)

    final_out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=final_out)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss=['binary_crossentropy'], optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def get_wordcnn_gru(text_vectorizer: TextVectorization) -> Model:
    """
    Get WordCNN-GRU model
    :param text_vectorizer: Text vectorizer adapted to a training data
    :return: Keras model
    """
    vocab_size = len(text_vectorizer.get_vocabulary())
    input_layer = Input(shape=(1,), dtype="string")
    x = text_vectorizer(input_layer)
    x = Embedding(vocab_size, 128)(x)
    num_cnn_filters = 128
    num_gru_filters = 512
    x3 = Conv1D(num_cnn_filters, 3, activation="tanh")(x)
    x3 = GlobalMaxPooling1D()(x3)

    x4 = Conv1D(num_cnn_filters, 4, activation="tanh")(x)
    x4 = GlobalMaxPooling1D()(x4)

    x5 = Conv1D(num_cnn_filters, 5, activation="tanh")(x)
    x5 = GlobalMaxPooling1D()(x5)

    conc = Concatenate()([x3, x4, x5])
    cnn = Dropout(0.2)(conc)

    gru = Bidirectional(GRU(num_gru_filters, activation="tanh"))(x)
    x = Dropout(0.2)(gru)

    conc = Concatenate()([cnn, gru])

    final_out = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=input_layer, outputs=final_out)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss=['binary_crossentropy'], optimizer=optimizer,
                  metrics=["accuracy"])
    return model
