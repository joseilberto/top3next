from math import floor
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from torch.optim.lr_scheduler import LambdaLR

import torch.nn as tnn
import torch as th
import numpy as np

from constants import GOOGLE_W2V

class bidirectional_LSTM(tnn.Module):
    def __init__(self, context_size, V, D, word2idx, hidden_nodes = 128,
                activation_last_layer = tnn.functional.log_softmax,     
                dropout = 0.2,
                learning_rate = 10**(-3),                
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


    def score_and_loss(self, X, Y, loss_fun, local, batch_size = 1024):
        n_batches = len(X) // batch_size
        if n_batches == 0:
            outputs = self.forward(X).view(len(X), -1)            
            logits = self.activation_last_layer(outputs, dim = 1)
            loss = loss_fun(logits, Y.long())
            if not local:
                loss = loss.get()                
            loss_value += loss.item()
            return self._score_from_outputs(outputs, Y.numpy()), loss_value
        accuracy = 0
        loss_value = 0
        for j in range(n_batches):
            X_batch = X[j*batch_size:(j*batch_size + batch_size)]
            Y_batch = Y[j*batch_size:(j*batch_size + batch_size)]
            outputs = self.forward(X_batch).view(batch_size, -1)
            logits = self.activation_last_layer(outputs, dim = 1)
            loss = loss_fun(logits, Y_batch.long())
            if not local:
                loss = loss.get()
            loss_value += loss.item()
            accuracy += self._score_from_outputs(outputs, Y_batch.numpy())
        return accuracy / n_batches, loss_value / n_batches

    
    def fit(self, X, Y, optimizer, batch_size = 64, epochs = 10, local = True,
            validation_split = 0.2, batch_print_epoch = 10):
        n_samples = len(X)
        X = th.tensor(X)        
        Y = th.tensor(Y)
        if validation_split:
            val_size = int(n_samples * (1 - validation_split))
            X_train, Y_train = X[:val_size], Y[:val_size]
            X_test, Y_test = X[val_size:], Y[val_size:]
        optimizer = optimizer(self.parameters(), lr = 1)
        n_batches = len(X_train) // batch_size
        step_size = 200
        clr = cyclical_lr(step_size)
        scheduler = LambdaLR(optimizer, [clr])
        batch_string = ("Running loss per batch epoch {}/{} on batch " + 
                        "{}/{}: {:.3f}\t Batch accuracy: {:.3f}").format
        epoch_string = ("Running loss for epoch {}/{}: {:.3f}\t" + 
                        "Training loss: {:.3f}\t" +
                        "Training accuracy: {:.3f}\t" + 
                        "Validation accuracy: {:.3f}\t" + 
                        "Validation loss: {:.3f}").format
        if not local:
            loss_fun = self.loss_fun.send(X.location)
        else:
            loss_fun = self.loss_fun        
        for epoch in range(epochs):
            running_loss = 0
            for j in range(n_batches):
                X_batch = X_train[j*batch_size:(j*batch_size + batch_size)]
                Y_batch = Y_train[j*batch_size:(j*batch_size + batch_size)]
                optimizer.zero_grad()                          
                outputs = self.forward(X_batch).view(batch_size, -1)                
                logits = self.activation_last_layer(outputs, dim = 1)
                accuracy = self._score_from_outputs(outputs, Y_batch.numpy())
                loss = loss_fun(logits, Y_batch.long())   
                loss.backward()
                scheduler.step()
                optimizer.step()
                if not local:
                    loss = loss.get()                
                running_loss += loss.item()
                if (j + 1) % batch_print_epoch == 0:
                    print(batch_string(epoch + 1, epochs, j + 1, n_batches, 
                                        running_loss / (j + 1), accuracy),
                                        end = "\r")   
            print("")
            train_accuracy, train_loss = self.score_and_loss(X_train, Y_train, 
                                                            loss_fun, local)
            validation_accuracy, validation_loss = self.score_and_loss(X_test, 
                                                    Y_test, loss_fun, local)
            print(epoch_string(epoch + 1, epochs, running_loss / n_batches,
                                train_loss, train_accuracy, 
                                validation_loss, validation_accuracy))



def cyclical_lr(stepsize, min_lr = 1e-5, max_lr = 1e-2):
    def relative(it, stepsize):
        cycle = floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    scaler = lambda x: 1.
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)    
    return lr_lambda


def embed_vocab(word2idx, V, D):
    word2vec = KeyedVectors.load_word2vec_format(GOOGLE_W2V, binary = True)
    embedding_matrix = np.zeros((V + 1, D))
    for word, i in word2idx.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec[word]
    return th.tensor(embedding_matrix, dtype = th.float32)


