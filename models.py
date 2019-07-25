from gensim.models import KeyedVectors
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD

import numpy as np

from constants import GOOGLE_W2V


def embed_vocab(vocabulary, V, D):
    word2vec = KeyedVectors.load_word2vec_format(GOOGLE_W2V, binary = True)
    embedding_matrix = np.zeros((V + 1, D))
    for word, i in vocabulary.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec[word]
    return embedding_matrix
    

def SPARSE_CATEGORICAL_CROSSENTROPY(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def LSTM_model(vocabulary, D = 300, context_size = 5, lr = 1e-2):
    V = len(vocabulary)
    embedding_matrix = embed_vocab(vocabulary, V, D)
    optimizer = SGD(lr = lr, nesterov = True)
    model = Sequential()
    model.add(Embedding(
                    input_dim = V + 1, output_dim = D, 
                    input_length = context_size,
                    weights=[embedding_matrix],
                    trainable=False)
                    )
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(V, activation = "linear"))    
    model.compile(loss = SPARSE_CATEGORICAL_CROSSENTROPY, optimizer = optimizer)
    return model
