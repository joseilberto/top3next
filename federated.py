import copy
import matplotlib.pyplot as plt
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


def update_history(history, new_history, iteration):
    for key, value in new_history.items():    
        padded = np.zeros(history[key].shape[0])
        value = np.array(value)            
        padded[:value.shape[0]] = value
        history[key] += (padded - history[key]) / iteration
    return history


def update_best_metrics(best_history, history, iteration):
    for key, value in history.items():
        best_history[key] += (value[-1] - best_history[key]) / iteration
    return best_history
    

def update_params(params_dict, new_model, iteration):
    for key, value in new_model.state_dict().items():
        if key in params_dict.keys():
            params_dict[key] += (value - params_dict[key]) / iteration
    return params_dict    


def train_federated(model, dataset, model_file, batch_size = 16,
                    epochs = 50, validation_split = 0.2):
    optimizer = th.optim.RMSprop    
    params_dict = {key: value for key, value in model.state_dict().items()}
    history = {
                "train_acc": np.zeros(epochs),
                "topk_train_acc": np.zeros(epochs),
                "train_loss": np.zeros(epochs), 
                "val_acc": np.zeros(epochs), 
                "topk_val_acc": np.zeros(epochs),
                "val_loss": np.zeros(epochs),
                }
    best_history = {
                "train_acc": 0,
                "topk_train_acc": 0,
                "train_loss": 0,
                "val_acc": 0,
                "topk_val_acc": 0,
                "val_loss": 0,
    }
    worker_string = "Copied model from worker {}/{}".format
    n_workers = len(dataset.datasets)
    for idx, (_, basedataset) in enumerate(dataset.datasets.items()):
        user_model = copy.deepcopy(model)
        X = basedataset.data
        Y = basedataset.targets
        user_model = user_model.send(X.location)
        user_history = user_model.fit(X, Y, optimizer, batch_size, epochs, 
                        local = False, validation_split = validation_split, 
                        verbose = False, topk_pred = 3)
        best_history = update_best_metrics(best_history, history, idx + 1)
        history = update_history(history, user_history, idx + 1)                        
        user_model = user_model.get()        
        params_dict = update_params(params_dict, user_model, idx + 1)
        end_string = "\n" if idx == n_workers - 1 else "\r"
        print(worker_string(idx + 1, n_workers), end = end_string)
        del user_model
    import ipdb; ipdb.set_trace()
    model.load_state_dict(params_dict, strict = False)
    return model, best_history


def train_multiple_federated(model, data, model_file, V, user_batch_size = 5, 
                            batch_size = 16, context_size = 5, epochs = 30,
                            validation_split = 0.2):
    unique_users = data.user.unique()
    n_user_batches = len(unique_users) // user_batch_size
    federated_model = copy.deepcopy(model)    
    for i in range(n_user_batches):
        cur_users = unique_users[i*user_batch_size:(i + 1)*user_batch_size]
        federated_dataset = get_federated_dataset(data, cur_users, context_size)
        federated_model, split_history = train_federated(federated_model, 
                                            federated_dataset, model, 
                                            batch_size, epochs, 
                                            validation_split)
        


