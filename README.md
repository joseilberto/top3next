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
- [X] Try a few different models for next word prediction respecting the 20ms time response for next word prediction in the server-side's data (all the results showed below are for 3 epochs of training):

Model | Top-1 Validation Score | Top-3 Validation Score
------------- | ------------- | -------------
GRU | 0.01 | 0.3
LSTM | 0.02 | 0.04
[genCNN](https://pdfs.semanticscholar.org/8645/643ad5dfe662fa38f61615432d5c9bdf2ffb.pdf) | 0.03 | 0.04
Bidirectional LSTM | 0.05 | 0.07

- [X] Train the server-side's model and send it to the batches of users;
- [X] Execute rounds of training into batches of users data;
- [X] Create a basic vizualiation pipeline for the federated training process;
- [ ] Observe the improvement of accuracy over rounds for the federated trained model.

## Datasets



