from local import train_local_model
from utils import extend_data, get_local_and_remote_data, merge_and_index_data
from models import LSTM_model


if __name__ == "__main__":
    data_file = "data/kaggle_twitter.csv"
    model_file = "data/LSTM_model_local.h5"
    min_tweets = 20
    local_share = 0.2
    context_size = 5
    data, tokenizer = merge_and_index_data(data_file, min_tweets)
    local_data, remote_data = get_local_and_remote_data(data, local_share)
    model = LSTM_model(tokenizer.word_index, context_size = context_size)
    train_local_model(model, local_data, tokenizer, model_file, 
                    context_size = context_size, retrain = True, epochs = 35)
