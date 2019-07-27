from tensorflow.keras.models import clone_model
from scipy.special import softmax

import syft as sy
import tensorflow as tf

from utils import extend_data

def train_federated(model, X, Y, model_file, batch_size = 16,
                    epochs = 50, validation_split = 0.1):
    hook = sy.KerasHook(tf.keras)
    base_host = 'localhost:4{:03}'.format
    worker = sy.TFEWorker(host = base_host(10))
    worker2 = sy.TFEWorker(host = base_host(11))
    worker3 = sy.TFEWorker(host = base_host(12))
    cluster = sy.TFECluster(worker, worker2, worker3)
    cluster.start()
    model.share(cluster)


def train_multiple_federated(model, data, model_file, user_split = 0.1, 
                            batch_size = 16, context_size = 5, epochs = 50,
                            validation_split = 0.1):
    unique_users = data.user.unique()
    user_batch_size = int(unique_users.shape[0] // (user_split*10**2))
    n_user_batches = int(1 // user_split)
    for i in range(n_user_batches):
        cur_users = unique_users[i*user_batch_size:(i + 1)*user_batch_size]
        for user in cur_users:
            federated_model = clone_model(model)
            federated_model.set_weights(model.get_weights())
            cur_data = data[data.user == user]
            X, Y = extend_data(cur_data.X, cur_data.Y, context_size)
            federated_model = train_federated(federated_model, X, Y, model_file, 
                        batch_size = batch_size, epochs = epochs, 
                        validation_split = validation_split)


