import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

def plot_loss_curves(history):
    """
    Plot the loss curves from the training history.
    
    Parameters:
    history: The training history.
    """
    try:
        logging.info("Plotting loss curves...")
        pd.DataFrame(history.history).plot()
        plt.title("Training curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss/Accuracy")
        plt.show()
    except Exception as e:
        logging.error("Error in plotting loss curves: %s", e)
        raise

def plot_learning_rate_vs_loss(history):
    """
    Plot learning rate vs. loss.
    
    Parameters:
    history: The training history.
    """
    try:
        logging.info("Plotting learning rate vs. loss...")
        lrs = 1e-5 * (10 ** (np.arange(len(history.history["loss"]))/20))
        plt.figure(figsize=(10, 7))
        plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss")
        plt.show()
    except Exception as e:
        logging.error("Error in plotting learning rate vs. loss: %s", e)
        raise