
from sklearn.metrics import accuracy_score
import logging
import tensorflow as tf

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model performance.
    
    Parameters:
    model: The trained model.
    x_test (DataFrame): The test features.
    y_test (Series): The test labels.
    
    Returns:
    float: The accuracy of the model.
    """
    try:
        logging.info("Evaluating model...")
        y_preds = model.predict(x_test)
        y_preds = tf.round(y_preds)
        accuracy = accuracy_score(y_test, y_preds)
        logging.info(f"Model accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        logging.error("Error in model evaluation: %s", e)
        raise