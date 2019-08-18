# Top3Next: A Gboard knock-off #

This repository is dedicated to the Project Showcase Challenge from Udacity 
Private AI Scholarship Program. Here, a small project is presented with
a simpler implementation of the functionalities of Gboard using Federated 
Learning for next word prediction. The original results were described in the paper 
[FEDERATED LEARNING FOR MOBILE KEYBOARD PREDICTION](https://arxiv.org/pdf/1811.03604.pdf)
by Google AI team. 

## Objectives

The present project intends to reproduce actual user data
that can be sent to a Machine Learning model. In order to have a smoother 
training and a better federated model the following steps in processing the dataset
were taken:

- [X] Restricting the data only to frequent users of the platform;
- [X] Clear the raw entry data from users (eliminating empty samples, 
duplicates and emoticons, ravelling contracted forms and removing stop words);
- [X] Split the data into server-side's and client-side's;
- [X] Transform text to sequences using a context of given size.

The processed data was then used in both the traditional supervised learning in
server-side and using federated learning in client-side reproduced using 
[Pysyft](https://github.com/OpenMined/PySyft). In order to find the best model
limited by a 20ms time response for each sample, the following actions were 
done:

- [X] Try a few different models for next word prediction in the server-side's 
data (all the results showed below are for 5 epochs of training):

Model | Top-1 Validation Score | Top-3 Validation Score
------------- | ------------- | -------------
GRU | 0.01 | 0.03
LSTM | 0.02 | 0.04
[genCNN](https://pdfs.semanticscholar.org/8645/643ad5dfe662fa38f61615432d5c9bdf2ffb.pdf) | 0.03 | 0.04
Bidirectional LSTM | 0.05 | 0.07

- [X] Train the server-side's model and send it to the batches of users;
- [X] Execute rounds of training into batches of users data;
- [X] Create a simple visualization pipeline for the federated training process;
- [ ] Observe the improvement of accuracy over rounds for the federated trained model.

## Datasets

I would like to acknowledge Alec Go, Richa Bhayani, and Lei Huang for 
making available the [sentiment140 dataset](http://help.sentiment140.com/for-students)
with 1,600,000 tweets. The choice of using twitter data is that it is the closest
one could assume that typed data looks like for Gboard application. This assumption
comes from the fact that most people use their smartphones to write friends and post
in social media.

Also, I would like to thank Google for the pre-trained word vectors from
[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/).
It allowed that the training process focused only on the neural network itself,
leaving the word embeddings unchanged during both the server-side and user-side's
training.

## Dependencies

All the dependencies are available in the [requirements file](requirements.yml) 
for [Anaconda](https://www.anaconda.com/distribution/#download-section). They
can be simply installed and sourced using the commands below:

```
conda env create -f environment.yml
conda activate top3next
```

## Data pre-processing

Since we are using sentiment140 dataset which has many instances of empty text samples,
duplicates, contractions, typos, etc, a data pre-processing was required. The 
pre-processing was inspired in two kernels from Kaggle 
([Paolo Ripamonti's](https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis)
and [Mohamed Gamal's](https://www.kaggle.com/gemyhamed/sentiment-analysis-word-embedding-lstm-cnn)). 
When put together both kernels, all the listed problems above are taken into
account and it was also added the stop words to avoid any unnecessary 
words to be added in our vocabulary.

A quick example of what the pre-processing can be seen below:
```python
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from utils.processors import get_cleaned_text

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
sentence = "the quick-brown-fox jumps over the lazy dog"
get_cleaned_text("the quick-brown-fox jumps over the lazy dog", stop_words, stemmer)
```

```
'quick brown fox jumps lazy dog'
```