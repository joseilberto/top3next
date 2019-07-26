from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import os

from models import CyclicLR, SPARSE_CATEGORICAL_CROSSENTROPY


def train_local_model(model, data, tokenizer, model_file, batch_size = 256, 
                context_size = 5, epochs = 25, validation_split = 0.2,
                retrain = False, show_progress = True):
    if os.path.isfile(model_file) and not retrain:
        return    
    if os.path.isfile(model_file):
        del model
        model = load_model(model_file, custom_objects = {
            "SPARSE_CATEGORICAL_CROSSENTROPY": SPARSE_CATEGORICAL_CROSSENTROPY
            })    
    X, Y = extend_data(data.X, data.Y, context_size)
    clr_triangular = CyclicLR(mode='triangular2', step_size = 2500)    
    progress = model.fit(X, Y, batch_size = batch_size, 
                callbacks = [clr_triangular],
                epochs = epochs, shuffle = False,
                validation_split = validation_split)    
    model.save(model_file)
    if show_progress:
        fig, (ax1, ax2) = plt.subplots(nrows = 2)
        ax1.set_title("Loss over epochs")
        ax1.plot(progress.history["loss"], c = "k", label = "Training Loss")
        ax1.plot(progress.history["val_loss"], c = "r", label = "Validation loss")
        ax1.legend()
        ax2.set_title("Accuracy over epochs")
        ax2.plot(progress.history["sparse_categorical_accuracy"], c = "k", 
                                            label = "Training Accuracy")
        ax2.plot(progress.history["val_sparse_categorical_accuracy"], c = "r", 
                                            label = "Validation Accuracy")
        ax2.legend()
        fig.savefig("./data/progress_plot.pdf", format = "pdf")
        plt.close()