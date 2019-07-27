from gensim.models import KeyedVectors
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM, Embedding
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, SGD

import numpy as np

from constants import GOOGLE_W2V

class CyclicLR(Callback):

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


def compile_model(model, lr):
    optimizer = Adam(lr = lr)
    model.compile(loss = SparseCategoricalCrossentropy(from_logits = True), 
                    optimizer = optimizer,
                    metrics = ["sparse_categorical_accuracy"])
    return model


def embed_vocab(word2idx, V, D):
    word2vec = KeyedVectors.load_word2vec_format(GOOGLE_W2V, binary = True)
    embedding_matrix = np.zeros((V + 1, D))
    for word, i in word2idx.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec[word]
    return embedding_matrix


def LSTM_model(word2idx, D = 300, context_size = 5, lr = 1e-3):
    V = len(word2idx) + 1
    embedding_matrix = embed_vocab(word2idx, V, D)
    model = Sequential()
    model.add(Embedding(
                    input_dim = V + 1, output_dim = D, 
                    input_length = context_size,
                    weights=[embedding_matrix],
                    trainable=False)
                    )
    model.add(Bidirectional(LSTM(128)))    
    model.add(Dropout(rate = 0.2))    
    model.add(Dense(V, activation = "linear"))    
    model = compile_model(model, lr)
    return model


