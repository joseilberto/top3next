# Top3Next: A Gboard knock-off #

This repository is dedicated to the Project Showcase Challenge from Udacity 
Private AI Scholarship Program. Here, a small project is presented with
a simpler implementation of the functionalities of Gboard using Federated 
Learning for next word prediction. The original results were described in the paper 
[FEDERATED LEARNING FOR MOBILE KEYBOARD PREDICTION](https://arxiv.org/pdf/1811.03604.pdf)
by Google AI team. 

The main objectives of the present project are listed below:
- [X] Restricting the data only to frequent users of the platform;
- [X] Clear the raw entry data from users (eliminating empty samples and duplicates, ravelling contracted forms and removing stop words);
- [X] Split the data into server-side's and client-side's;
- [X] Transform text to sequences;
- [X] Try a few different models for next word prediction respecting the 20ms time response for next word prediction in the server-side's data (all the results showed below are for 3 epochs of training in ):

Model | Top-1 Score ~ 0.5
------------- | -------------
GRU | 0.5
LSTM | 0.7
[genCNN](https://pdfs.semanticscholar.org/8645/643ad5dfe662fa38f61615432d5c9bdf2ffb.pdf) | 0.7
Bidirectional LSTM | 0.11


## Datasets


