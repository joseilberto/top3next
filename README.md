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
- [X] Preserve the timestamp of the usage and keep data ordered by the respective timestamp;
- [X] Split the data into server-side's and client-side's;
- [X] Transform text to sequences using a context of given size.

The processed data was then used in both the traditional supervised learning in
server-side and using federated learning in client-side reproduced using 
[Pysyft](https://github.com/OpenMined/PySyft). In order to find the best model
limited by a 20ms time response for each sample, the following actions were 
done:

- [X] Try a few different models for next word prediction in the server-side 
data;
- [X] Train the server-side model and send it to the batches of users;
- [X] Execute rounds of training into batches of users data;
- [X] Create a simple visualization pipeline for the federated training process;
- [ ] Observe the improvement of accuracy over rounds for the federated trained model.

## Datasets

I would like to acknowledge Alec Go, Richa Bhayani, and Lei Huang for 
making available the [sentiment140 dataset](http://help.sentiment140.com/for-students)
with 1,600,000 tweets. The choice of using Twitter data can be justified by the
fact that it is the closest one could assume that typed data looks like for 
Gboard application. This assumption comes from the fact that most people use 
their smartphones to write friends and post in social media.

Also, I would like to thank Google for the word2vec pre-trained word vectors from
[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/).
It allowed that the training process focused only on the neural network itself,
leaving the word embeddings unchanged during both the server-side and user-side
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

A quick example of what the pre-processing does can be seen below:
```python
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from utils.processors import get_cleaned_text

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
sentence = "the quick-brown-fox jumps over the lazy dog"
cleaned_sentence = get_cleaned_text("the quick-brown-fox jumps over the lazy dog", 
                    stop_words, stemmer)
print(cleaned_sentence)
```

```console
'quick brown fox jumps lazy dog'
```

Other than this pre-processing, we make sure that the data is kept ordered by
its timestamp, implying that it will be processed by the Neural Network in FIFO
order. It was chosen in such a manner, because language usually trends in such
a fashion that the observed patterns for a user A may influence user B to use
not only the same words, but also the same structures. By keeping
the data organized preserving the actual time it was stored, we hope that 
it improves the odds that the Neural Network learns trends of specific words 
given some contexts.

## Model Selection and server training

Regarding the model selection, a few models were tested before deciding on using
the bidirectional LSTM. We can observe a table below listing some results for
5 epochs of training in server-side data for 4 models that were tested:

Model | Top-1 Train score | Top-1 Validation Score | Top-3 Train Score | Top-3 Validation Score 
------------- | ------------- | ------------- | ------------- | -------------
GRU | 0.07 | 0.01 | 0.15 | 0.03 
LSTM | 0.09 | 0.03 | 0.20 | 0.05 
[genCNN](https://pdfs.semanticscholar.org/8645/643ad5dfe662fa38f61615432d5c9bdf2ffb.pdf) | 0.10 | 0.03 | 0.20 | 0.04 
Bidirectional LSTM | 0.14 | 0.05 | 0.23 | 0.07 

It is important to notice that in all cases the trained models could not
achieve the accuracy reported in Gboard paper if considered a validation dataset 
within each user data. These results suggest that in all scenarios, even though
the train loss and validation loss are similar, the models overfit and 
tend to find better predictions only in the data it is being trained. 

### Visualizing the predictions

In order to understand how these accuracies relate to word predictions, the
following code brings a simple example of how we can visualize these predictions.

```python
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import torch as th

from utils.models import bidirectional_LSTM
from utils.processors import get_cleaned_text, print_predictions

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
context_size = 5
D = 300
word2idx = ... #word2idx dictionary
hidden_nodes = 128
local_model = "local_model.pth" #local pre-trained model file
model = bidirectional_LSTM(context_size, len(word2idx), D, word2idx, hidden_nodes)
model.load_state_dict(th.load(model_file))
sentence_example = "i have a colossal headache. it  feels like a nuclear weapon testing facility in there"
cleaned_sentence = get_cleaned_text(sentence_example, stop_words, stemmer).split()
print(" ".join(cleaned_sentence))
print_predictions(cleaned_sentence, model)
```

```console
'colossal headache feels like nuclear weapon testing facility'
Previous word: colossal          Expected word: headache         Predictions: day           headache        piece
Previous word: headache          Expected word: feels            Predictions: throat        hope            morning
Previous word: feels             Expected word: like             Predictions: like          better          guilty
Previous word: like              Expected word: nuclear          Predictions: good          title           day
Previous word: nuclear           Expected word: weapon           Predictions: browser       4th             articles
Previous word: weapon            Expected word: testing          Predictions: balm          question        side
```

For this specific case, the top-3 accuracy is 0.33 what is expected for a sample 
contained in the training data.


## Federated Learning results

The Federated Learning required more steps than a traditional supervised
learning approach would require in this scenario. There are multiple factors
that distinguish the paradigms, two of them are strictly related to how and when
we should update the model and send to new users. 

Basically, we consider the 4 main concepts that help us to define when update
the model and how much each user contributes to the model:

- Round: A time step that when finished the federated model is updated;
- Users batch size: The number of users that is required to complete a round and update the model;
- Users sample size: Different users have different keyboard usage, being some
more active than others. In order to make them all have the same weight, we 
reduced the sample size of each user to its bare-minimum (since each user is required to have
at least a minimum amount of tweets in order to be included in the dataset). 
- Early stopping: Since each user may train faster than what was possible with
the local setup, we can impose an early stopping mechanism to avoid overfitting
for each user which would represent poor models in general.

Put all these factors together, we were expected to find that the federated
model performed just as good as the local model, but consuming less server time
and improving the overall prediction for each individual user. 

.. raw:: html

<script> src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<div>\n        \n        \n            <div id="abbc95cb-5fd0-4671-afd1-0e46358d1a2b" class="plotly-graph-div" style="height:100%; width:100%;"></div>\n            <script type="text/javascript">\n                \n                    window.PLOTLYENV=window.PLOTLYENV || {};\n                    \n                if (document.getElementById("abbc95cb-5fd0-4671-afd1-0e46358d1a2b")) {\n                    Plotly.newPlot(\n                        \'abbc95cb-5fd0-4671-afd1-0e46358d1a2b\',\n                        [{"mode": "lines+markers", "name": "Train Accuracy", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "y": [0.1386759130048189, 0.15252045886186585, 0.13960859059546105, 0.09635905579671618, 0.12800705109873708, 0.12672347573359422, 0.11248255934850188, 0.07846968556544934, 0.10761383784059644, 0.06940059481459926, 0.09117126226101946, 0.044978263510034255, 0.04375513731556641, 0.05845076201864845, 0.04099392040769076]}, {"mode": "lines+markers", "name": "Validation Accuracy", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "y": [0.031149085033068687, 0.03436469732203444, 0.03243371215976838, 0.03918583945400494, 0.0434237166281152, 0.02563092691097639, 0.02866239797818745, 0.016435502257117658, 0.025846990163840587, 0.02268184560867488, 0.02998681630817243, 0.008565203859321507, 0.02336434902224376, 0.010332470109255823, 0.023514899372284768]}],\n                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Federated Learning accuracy evolution"}, "xaxis": {"title": {"text": "Round"}}, "yaxis": {"title": {"text": "Accuracy"}}},\n                        {"responsive": true}\n                    )\n                };\n                \n            </script>\n        </div>
