import logging
import tensorflow as tf
import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_preprocess_data
from src.eda import perform_eda
from src.evaluation import evaluate_model
from src.model_training import create_model, compile_model, train_model
from src.visualization import plot_loss_curves, plot_learning_rate_vs_loss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("employee_attrition.log"),
    logging.StreamHandler()
])

def main():
    try:
        # Step 1: Load and preprocess data
        x_train, x_test, y_train, y_test = load_and_preprocess_data('H:/My Drive/BISI II/Data Science/Term Assignments/Deep_Learning_with_Tensorflow/data/employee_attrition.csv')
        
        # Step 2: Perform EDA
        df = pd.read_csv('H:/My Drive/BISI II/Data Science/Term Assignments/Deep_Learning_with_Tensorflow/data/employee_attrition.csv')
        perform_eda(df)
        
        # Step 3: Create and compile the model
        model = create_model([
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model = compile_model(model, learning_rate=0.001)
        
        # Step 4: Train the model
        history = train_model(model, x_train, y_train, epochs=100)
        
        # Step 5: Evaluate the model
        accuracy = evaluate_model(model, x_test, y_test)
        logging.info(f"Model accuracy: {accuracy}")
        
        # Step 6: Visualize the results
        plot_loss_curves(history)
        plot_learning_rate_vs_loss(history)
        
    except Exception as e:
        logging.error("Error in main function: %s", e)
        raise

if __name__ == '__main__':
    main()