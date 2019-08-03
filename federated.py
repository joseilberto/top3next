import copy
import numpy as np
import syft as sy
import torch as th

from utils import extend_data


def get_federated_dataset(data, users, context_size):
    users_data = []
    hook = sy.TorchHook(th)
    for user in users:
        user_worker = sy.VirtualWorker(hook, id = user)
        cur_data = data[data.user == user]
        X, Y = extend_data(cur_data.X, cur_data.Y, context_size)
        X = th.tensor(X)
        Y = th.tensor(Y)
        users_data.append(sy.BaseDataset(X, Y).send(user_worker))
    return sy.FederatedDataset(users_data)


def train_federated(model, dataloader, model_file, batch_size = 16,
                    epochs = 50, validation_split = 0.1):
    optimizer = th.optim.RMSprop
    model.fit_generator(dataloader, optimizer, batch_size, epochs,
                            local = False, validation_split = 0.2,
                            batch_print_epoch = 2)
    import ipdb; ipdb.set_trace()

    


def train_multiple_federated(model, data, model_file, V, user_split = 0.1, 
                            batch_size = 16, context_size = 5, epochs = 50,
                            validation_split = 0.1):
    unique_users = data.user.unique()
    user_batch_size = int(unique_users.shape[0] // (user_split*10**2))
    n_user_batches = int(1 // user_split)
    federated_model = copy.deepcopy(model)    
    for i in range(n_user_batches):
        cur_users = unique_users[i*user_batch_size:(i + 1)*user_batch_size]
        federated_dataset = get_federated_dataset(data, cur_users, context_size)
        federated_loader = sy.FederatedDataLoader(federated_dataset,
                                    shuffle = False, batch_size = batch_size)
        federated_model = train_federated(federated_model, federated_loader, 
                                            model)


