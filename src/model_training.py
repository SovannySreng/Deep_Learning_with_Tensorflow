
import tensorflow as tf
import logging

def create_model(layers):
    """
    Create a TensorFlow model.
    
    Parameters:
    layers (list): List of tf.keras.layers to add to the model.
    
    Returns:
    model: The created TensorFlow model.
    """
    try:
        logging.info("Creating model...")
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential(layers)
        logging.info("Model created.")
        return model
    except Exception as e:
        logging.error("Error in model creation: %s", e)
        raise

def compile_model(model, learning_rate=0.001):
    """
    Compile the TensorFlow model.
    
    Parameters:
    model: The TensorFlow model to compile.
    learning_rate (float): The learning rate for the optimizer.
    
    Returns:
    model: The compiled TensorFlow model.
    """
    try:
        logging.info("Compiling model...")
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                      metrics=['accuracy'])
        logging.info("Model compiled.")
        return model
    except Exception as e:
        logging.error("Error in model compilation: %s", e)
        raise

def train_model(model, x_train, y_train, epochs=50):
    """
    Train the TensorFlow model.
    
    Parameters:
    model: The TensorFlow model to train.
    x_train (DataFrame): The training features.
    y_train (Series): The training labels.
    epochs (int): The number of epochs to train the model.
    
    Returns:
    history: The training history.
    """
    try:
        logging.info("Training model...")
        history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
        logging.info("Model trained.")
        return history
    except Exception as e:
        logging.error("Error in model training: %s", e)
        raise