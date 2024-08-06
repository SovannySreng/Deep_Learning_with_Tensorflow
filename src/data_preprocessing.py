

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

def load_and_preprocess_data(file_path='H:/My Drive/BISI II/Data Science/Term Assignments/Deep_Learning_with_Tensorflow/data/employee_attrition.csv'):
    """
    Load and preprocess the dataset.
    
    Parameters:
    file_path (str): The path to the dataset file.
    
    Returns:
    x_train, x_test, y_train, y_test: The train-test split data.
    """
    try:
        logging.info("Loading dataset...")
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded.")
        
        # Separating target variable and other variables
        Y = df.Attrition
        X = df.drop(columns=['Attrition'])
        
        # Scaling the data
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Splitting the data
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
        logging.info("Data preprocessing completed.")
        
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("Error in data preprocessing: %s", e)
        raise