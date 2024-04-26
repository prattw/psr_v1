import sys
import logging
import numpy as np  # Ensure numpy is imported
from tensorflow import keras
sys.path.append('../src')  # Adjust this path as needed

from data_preparation import load_data, preprocess_and_save_data, create_sequences

# Set up logging
logging.basicConfig(filename='model_evaluation.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load the saved model
model_path = '/Users/williampratt/Library/Mobile Documents/com~apple~CloudDocs/Documents/project_sea_ranch/models/lstm_prediction_model_v1.keras'  # Update this path to where your model is saved
try:
    model = keras.models.load_model(model_path)
except FileNotFoundError:
    logging.error("Model file not found.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit(1)

def load_and_preprocess(stock_file_path):
    test_data = load_data(stock_file_path)
    if test_data is None:
        logging.error(f"Failed to load test data from {stock_file_path}")
        return None
    preprocessed_test_data = preprocess_and_save_data(test_data)
    if preprocessed_test_data is None:
        logging.error(f"Failed to preprocess data from {stock_file_path}")
    return preprocessed_test_data

# Paths to test data files
spy_file_path = '/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/SPY_test_data.csv'
voo_file_path = '/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/VOO_test_data.csv'
nvda_file_path = '/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/NVDA_test_data.csv'

# Load and preprocess the data for each stock
preprocessed_spy_data = load_and_preprocess(spy_file_path)
preprocessed_voo_data = load_and_preprocess(voo_file_path)
preprocessed_nvda_data = load_and_preprocess(nvda_file_path)

def evaluate_stock_model(model, preprocessed_data, sequence_length=29, num_features=5):
    if preprocessed_data is None:
        return None
    test_sequences = create_sequences(preprocessed_data.values, sequence_length=29, num_features=5)
    print("Shape of test sequences for stock:", test_sequences.shape)
    test_features = test_sequences
    print("Shape of test features for stock:", test_features.shape)
    test_labels = np.array([seq[-1, -1] for seq in test_sequences])  # Modify as per your label structure
    print("Shape of test labels for stock:", test_labels.shape)
    performance = model.evaluate(test_features, test_labels)
    return performance

# Evaluate the model for each stock
stocks_evaluation = {}
for stock, data in [('SPY', preprocessed_spy_data), ('VOO', preprocessed_voo_data), ('NVDA', preprocessed_nvda_data)]:
    print(f"Evaluating model for {stock}")
    stock_performance = evaluate_stock_model(model, data)
    if stock_performance is not None:
        stocks_evaluation[stock] = stock_performance
        print(f"{stock} Model Evaluation:", stock_performance)