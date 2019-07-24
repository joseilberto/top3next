from utils import merge_and_index_data

if __name__ == "__main__":
    data_file = "data/kaggle_twitter.csv"
    min_tweets = 20
    data, tokenizer = merge_and_index_data(data_file, min_tweets)
    
