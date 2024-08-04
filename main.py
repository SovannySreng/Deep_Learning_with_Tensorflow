
from src.data_preprocessing import load_data, preprocess_data
from src.eda import eda
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.visualization import plot_histograms, plot_categorical_distribution
from src.utils import setup_logging, log_error

def main():
    setup_logging()
    
    try:
        df = load_data('H:/My Drive/BISI II/Data Science/Term Assignments/Deep_Learning_with_Tensorflow/data/employee_attrition.csv')  # Use relative path
        
        # Perform EDA
        eda(df)
        
        # Preprocess Data
        df = preprocess_data(df)
        
        # Feature Engineering
        df = feature_engineering(df)
        
        # Visualizations
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        plot_histograms(df, num_cols)
        plot_categorical_distribution(df, cat_cols)
        
        # Prepare data for training
        X = df.drop(columns=['Attrition'])  # Adjust according to your feature columns
        y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Train the model
        model = train_model(X, y)
        
        # Evaluate the model
        evaluate_model(model, X, y)
        
    except Exception as e:
        log_error(e)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()