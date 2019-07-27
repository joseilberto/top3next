from federated import train_federated, train_multiple_federated
from local import train_local_model
from utils import get_local_and_remote_data, merge_and_index_data
from models import LSTM_model




if __name__ == "__main__":
    data_file = "data/kaggle_twitter.csv"
    model_file = "data/LSTM_model_local2.h5"
    federated_file = "data/LSTM_model_federated.h5"
    dump_file = "data/pandas_df.pkl"
    word2idx_file = "data/tokenizer_keys.pkl"
    min_tweets = 20
    local_share = 0.2
    context_size = 5
    epochs = 10
    data, word2idx = merge_and_index_data(data_file, dump_file, word2idx_file, min_tweets)
    local_data, remote_data = get_local_and_remote_data(data, local_share)
    model = LSTM_model(word2idx, context_size = context_size)
    model = train_local_model(model, local_data, model_file, 
                    context_size = context_size, epochs = epochs)
    train_multiple_federated(model, remote_data, federated_file, len(word2idx))
