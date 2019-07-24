from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import nltk
import pandas as pd
import re



def get_cleaned_text(text, stem = False):
    cleaning_regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    text = re.sub(cleaning_regex, " ", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
                continue
            tokens.append(token)
    return " ".join(tokens)


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
    data = pd.read_csv(input_file, encoding = "ISO-8859-1", header = None, 
                            names = columns)
    data.drop(["target", "flag", "ids"], axis = 1, inplace = True)
    users = data.groupby(by = "user").apply(len) > min_tweets
    data = data[data.user.isin(users[users].index)]
    data = index_data_by_date(data)
    data.text = data.text.apply(lambda x: get_cleaned_text(x))
    sequences, tokenizer = text_to_sequence(data.text, Tokenizer)
    data["sequence"] = sequences
    return data, tokenizer
    

def text_to_sequence(texts, tokenizer):
    tokenizer = tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer.texts_to_sequences(texts), tokenizer
