from utils import get_local_and_users_data, merge_and_index_data
from models import LSTM_model


def train_model(model, data, tokenizer, validation_split = 0.2, epochs = 10):
    pass




if __name__ == "__main__":
    data_file = "data/kaggle_twitter.csv"
    model_file = "data/LSTM_model.h5"
    min_tweets = 20
    local_share = 0.2
    data, tokenizer = merge_and_index_data(data_file, min_tweets)
    local_data, users_data = get_local_and_users_data(data, local_share)
    model = LSTM_model(tokenizer.word_index)
    train_model(model, data, tokenizer)
