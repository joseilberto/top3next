from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_sequence, PackedSequence

import torch.nn as tnn
import torch as th
import numpy as np

from constants import GOOGLE_W2V

class bidirectional_LSTM(tnn.Module):
    def __init__(self, context_size, V, D, word2idx, hidden_nodes = 128,
                activation_last_layer = tnn.functional.log_softmax,     
                dropout = 0.2,
                learning_rate = 10**(-2),                
                loss_fun = tnn.functional.nll_loss):                            
        super(bidirectional_LSTM, self).__init__()        
        self.context_size = context_size
        self.V = V        
        self.D = D
        self.word2idx = word2idx
        self.hidden_nodes = hidden_nodes        
        self.activation_last_layer = activation_last_layer        
        self.dropout = dropout
        self.lr = learning_rate
        self.loss_fun = loss_fun
        self._build()       


    def _build(self):                   
        self.embedding = tnn.Embedding.from_pretrained(embed_vocab(self.word2idx, 
                                    self.V, self.D), freeze = True)        
        self.bi_lstm = tnn.LSTM(input_size = self.D, 
                            hidden_size = self.hidden_nodes,
                            num_layers = 1, bias = True, batch_first = True,
                            dropout = self.dropout, bidirectional = True)
        self.fc_layer = tnn.Linear(2*self.hidden_nodes, self.V)

    
    def forward(self, X):
        embedded = self.embedding(X)
        output, hidden_state = self.bi_lstm(embedded)
        output = self.fc_layer(output)
        return output
    
    
    def _score_from_outputs(self, outputs, Y):
        preds = tnn.functional.softmax(outputs, dim = 1).topk(1, dim = 1).indices
        accuracy = np.mean(preds.numpy() == Y)
        return accuracy

    
    def fit(self, X, Y, optimizer, batch_size = 64, epochs = 10, local = True,
            batch_print_epoch = 100):
        n_samples = len(X)
        X = th.tensor(X)        
        Y = th.tensor(Y)
        optimizer = optimizer(self.parameters(), lr = self.lr)
        n_batches = n_samples // batch_size
        batch_string = ("Running loss per batch epoch {}/{} on batch " + 
                        "{}/{}: {:.3f}\t Batch accuracy: {:.3f}").format
        epoch_string = "Running loss for epoch {}/{}: {:.3f}".format
        if not local:
            loss_fun = self.loss_fun.send(X.location)
        else:
            loss_fun = self.loss_fun        
        for epoch in range(epochs):
            running_loss = 0
            for j in range(n_batches):
                X_batch = X[j*batch_size:(j*batch_size + batch_size)]
                Y_batch = Y[j*batch_size:(j*batch_size + batch_size)]
                optimizer.zero_grad()                          
                outputs = self.forward(X_batch).view(batch_size, -1)
                logits = self.activation_last_layer(outputs, dim = 1)
                accuracy = self._score_from_outputs(outputs, Y_batch.numpy())
                loss = loss_fun(logits, Y_batch.long())   
                loss.backward()         
                optimizer.step()
                if not local:
                    loss = loss.get()                
                running_loss += loss.item()
                if (j + 1) % batch_print_epoch == 0:
                    print(batch_string(epoch + 1, epochs, j + 1, n_batches, 
                                        running_loss / (j + 1), accuracy))                
            print(epoch_string(epoch + 1, epochs, running_loss / n_batches))


def embed_vocab(word2idx, V, D):
    word2vec = KeyedVectors.load_word2vec_format(GOOGLE_W2V, binary = True)
    embedding_matrix = np.zeros((V + 1, D))
    for word, i in word2idx.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec[word]
    return th.tensor(embedding_matrix, dtype = th.float32)


