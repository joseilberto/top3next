from math import floor
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from torch.optim.lr_scheduler import CyclicLR

import torch.nn as tnn
import torch as th
import numpy as np

from constants import GOOGLE_W2V

class bidirectional_LSTM(tnn.Module):
    def __init__(self, context_size, V, D, word2idx, hidden_nodes = 128,
                n_rnn_layers = 1,
                activation_last_layer = tnn.functional.log_softmax,     
                dropout = 0.2,
                learning_rate = 10**(-3),                
                loss_fun = tnn.NLLLoss()):                            
        super(bidirectional_LSTM, self).__init__()        
        self.context_size = context_size
        self.V = V        
        self.D = D
        self.word2idx = word2idx
        self.hidden_nodes = hidden_nodes
        self.n_fc_layer1 = 8*hidden_nodes        
        self.n_rnn_layers = n_rnn_layers
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
                            num_layers = self.n_rnn_layers, bias = True, 
                            batch_first = True, dropout = self.dropout, 
                            bidirectional = True)
        self.fc_layer = tnn.Linear(2*self.hidden_nodes, self.V)
        self.relu_layer = tnn.ReLU()

    
    def _init_hidden(self, batch_size):
        h0 = th.autograd.Variable(th.zeros(2*self.n_rnn_layers, batch_size, 
                                                        self.hidden_nodes))
        c0 = th.autograd.Variable(th.zeros(2*self.n_rnn_layers, batch_size, 
                                                        self.hidden_nodes))
        return h0, c0

    
    def forward(self, X, training = False):
        embedded = self.embedding(X)
        output, self.h_lstm = self.bi_lstm(embedded, self.h_lstm)                
        output = output.transpose(1, 2)                
        output = self.relu_layer(output[:, :, -1])
        output = self.fc_layer(output)
        return output
    
    
    def _score_from_outputs(self, outputs, Y):
        preds = tnn.functional.softmax(outputs, dim = 1).topk(1, dim = 1).indices
        accuracy = np.mean(preds.numpy().ravel().astype(np.int32) == Y)
        return accuracy


    def score_and_loss(self, X, Y, loss_fun, local, batch_size = 1024):
        n_batches = len(X) // batch_size
        if n_batches == 0:
            self.h_lstm = self._init_hidden(len(X))
            outputs = self.forward(X).view(len(X), -1)            
            logits = self.activation_last_layer(outputs, dim = 1)
            loss = loss_fun(logits, Y.long())
            if not local:
                loss = loss.get()                
            loss_value = loss.item()
            return self._score_from_outputs(outputs, Y.numpy()), loss_value
        accuracy = 0
        loss_value = 0
        for j in range(n_batches):
            self.h_lstm = self._init_hidden(batch_size)
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

    
    def fit_generator(self, generator, optimizer, batch_size = 64, epochs = 10,
                        local = True, validation_split = 0.2,
                        batch_print_epoch = 10):
        for X, Y in generator:
            if not local:
                self.send(X.location)
            self.fit(X, Y, optimizer, batch_size, epochs, local, 
                        validation_split, batch_print_epoch)
            if not local:
                self.get()            

    
    def fit(self, X, Y, optimizer, batch_size = 64, epochs = 10, local = True,
            validation_split = 0.2, batch_print_epoch = 1):        
        n_samples = len(X)
        if local:
            X = th.tensor(X)        
            Y = th.tensor(Y)
        if validation_split:
            val_size = int(n_samples * (1 - validation_split))
            X_train, Y_train = X[:val_size], Y[:val_size]
            X_test, Y_test = X[val_size:], Y[val_size:]
        optimizer = optimizer(self.parameters(), lr = self.lr)
        n_batches = len(X_train) // batch_size
        scheduler = CyclicLR(optimizer, self.lr / 1000, self.lr, 
                    step_size_up = 4000)
        batch_string = ("Running loss per batch epoch {}/{} on batch " + 
                        "{}/{}: {:.3f}\t Running accuracy: {:.3f}").format
        epoch_string = ("Running loss for epoch {}/{}: {:.3f}\t" + 
                        "Training loss: {:.3f}\t" +
                        "Training accuracy: {:.3f}\t" + 
                        "Validation loss: {:.3f}\t" + 
                        "Validation accuracy: {:.3f}").format
        if not local:            
            loss_fun = self.loss_fun.send(X.location)            
        else:
            loss_fun = self.loss_fun        
        for epoch in range(epochs):
            running_loss = 0
            running_acc = 0
            for j in range(n_batches):
                self.h_lstm = self._init_hidden(batch_size)
                if not local:
                    self.h_lstm = self.h_lstm.send(X.location)
                X_batch = X_train[j*batch_size:(j*batch_size + batch_size)]
                Y_batch = Y_train[j*batch_size:(j*batch_size + batch_size)]
                optimizer.zero_grad()                
                outputs = self.forward(X_batch, True)                
                logits = self.activation_last_layer(outputs, dim = 1)
                running_acc += self._score_from_outputs(outputs, Y_batch.numpy())
                loss = loss_fun(logits, Y_batch.long())   
                loss.backward()
                scheduler.step()
                optimizer.step()
                if not local:
                    loss = loss.get()                
                running_loss += loss.item()
                if (j + 1) % batch_print_epoch == 0:
                    print(batch_string(epoch + 1, epochs, j + 1, n_batches, 
                                        running_loss / (j + 1), 
                                        running_acc / (j + 1)),
                                        end = "\r")   
            print("")
            train_accuracy, train_loss = self.score_and_loss(X_train, Y_train, 
                                                            loss_fun, local)
            validation_accuracy, validation_loss = self.score_and_loss(X_test, 
                                                    Y_test, loss_fun, local)
            print(epoch_string(epoch + 1, epochs, running_loss / n_batches,
                                train_loss, train_accuracy, 
                                validation_loss, validation_accuracy))


def embed_vocab(word2idx, V, D):
    word2vec = KeyedVectors.load_word2vec_format(GOOGLE_W2V, binary = True)
    embedding_matrix = np.zeros((V + 1, D))
    for word, i in word2idx.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec[word]
    return th.tensor(embedding_matrix, dtype = th.float32)


