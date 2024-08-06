

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

def perform_eda(df):
    """
    Perform exploratory data analysis.
    
    Parameters:
    df (DataFrame): The input data.
    """
    try:
        logging.info("Performing EDA...")
        
        # Plot distributions
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols].hist(figsize=(14, 14))
        plt.show()
        
        # Plot categorical distributions
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=col, data=df)
            plt.title(f'Distribution of {col}')
            plt.show()
        
        logging.info("EDA completed.")
    except Exception as e:
        logging.error("Error in EDA: %s", e)
        raise