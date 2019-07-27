from collections import OrderedDict
from tensorflow.keras.models import clone_model
from scipy.special import softmax

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from models import compile_model
from utils import extend_data


def get_federated_dataset(data, users, context_size):
    users_data = {}
    for user in users:
        cur_data = data[data.user == user]
        X, Y = extend_data(cur_data.X, cur_data.Y, context_size)
        users_data[user] = {"x": X, "y": Y}
    return tff.simulation.FromTensorSlicesClientData(users_data)


def get_federated_model(model, example_dataset, V, context_size):
    federated_model = compile_model(clone_model(model), 1e-3)
    x = tf.constant(np.random.randint(1, V, size = [1, context_size]))
    y = tf.constant(np.random.randint(1, V, size = [1, 1]))
    dummy_batch = OrderedDict([('x', x), ('y', y)])
    return tff.learning.from_compiled_keras_model(federated_model, 
                                                dummy_batch = dummy_batch)

def train_federated(model, X, Y, model_file, batch_size = 16,
                    epochs = 50, validation_split = 0.1):
    pass


def train_multiple_federated(model, data, model_file, V, user_split = 0.1, 
                            batch_size = 16, context_size = 5, epochs = 50,
                            validation_split = 0.1):
    unique_users = data.user.unique()
    user_batch_size = int(unique_users.shape[0] // (user_split*10**2))
    n_user_batches = int(1 // user_split)
    model_copy = clone_model(model)
    model_copy.set_weights(model.get_weights())
    for i in range(n_user_batches):
        cur_users = unique_users[i*user_batch_size:(i + 1)*user_batch_size]
        federated_dataset = get_federated_dataset(data, cur_users, context_size)
        example_dataset = federated_dataset.create_tf_dataset_for_client(
            federated_dataset.client_ids[0]
        )
        federated_average = tff.learning.build_federated_averaging_process(
            model_fn = lambda: get_federated_model(model_copy, example_dataset, 
                                V, context_size) 
        )
        import ipdb; ipdb.set_trace()


