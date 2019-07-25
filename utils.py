from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from random import sample

import numpy as np
import pandas as pd
import re

from constants import CLEANING_REGEX, NEGATIONS_DICT


def get_cleaned_text(text, stop_words, stemmer, stem = False):    
    neg_pattern = re.compile(r'\b(' + '|'.join(NEGATIONS_DICT.keys()) + r')\b')    
    text = re.sub(CLEANING_REGEX, " ", str(text).lower()).strip()
    text = neg_pattern.sub(lambda x: NEGATIONS_DICT[x.group()], text)
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
                continue
            tokens.append(token)    
    text = " ".join(tokens)
    text = re.sub("n't", "not", text)
    return re.sub("'s", "is", text)


def get_local_and_users_data(data, local_share = 0.1):
    unique_users = data.user.unique()
    local_users = np.random.choice(unique_users, 
                                    int(local_share*unique_users.shape[0]))
    remote_users = ~np.isin(unique_users, local_users)
    local_data = data[data.user.isin(local_users)]
    remote_data = data[data.user.isin(remote_users)]
    #TODO Finish it later




def index_data_by_date(data, string_tz = "PDT"):
    timezone = 'US/Pacific' if "PDT" or "PT" in string_tz else "UTC"
    data.date = data.date.str.replace(string_tz, "")
    data.date = data.date.astype("datetime64[ns]")
    data.index = data.date
    data.drop(["date"], axis = 1, inplace = True)
    data.index = data.index.tz_localize(timezone)
    return data


def merge_and_index_data(input_file, min_tweets = 20):
    columns = ["target", "ids", "date", "flag", "user", "text"]
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    data = pd.read_csv(input_file, encoding = "ISO-8859-1", header = None, 
                            names = columns)
    data.drop(["target", "flag", "ids"], axis = 1, inplace = True)
    users = data.groupby(by = "user").apply(len) > min_tweets
    data = data[data.user.isin(users[users].index)]
    data = index_data_by_date(data)
    data.text = data.text.apply(lambda x: get_cleaned_text(x, stop_words, stemmer))
    data.drop_duplicates(subset = ["text"], keep = False, inplace = True)
    sequences, tokenizer = text_to_sequence(data.text, Tokenizer)
    data["sequence"] = sequences
    data = data[data.sequence.map(lambda x: len(x)) > 0]    
    data = data.merge(data.sequence.apply(lambda x: split_X_and_Y(x)), 
                    left_index = True, right_index = True)    
    return data, tokenizer


def split_X_and_Y(sequence):
    X = [0]    
    Y = [sequence[0]]
    for idx, token in enumerate(sequence[:-1]):
        X.append(token)
        Y.append(sequence[idx + 1])
    return pd.Series({"X": X, "Y": Y})


def text_to_sequence(texts, tokenizer):
    tokenizer = tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer.texts_to_sequences(texts), tokenizer


