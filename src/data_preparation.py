import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Enhanced logging setup
logging.basicConfig(level=logging.INFO, filename='data_preparation.log', filemode='w', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filename):
    logging.info(f"Attempting to load data from {filename}")
    if not os.path.exists(filename):
        logging.error(f"File not found: {filename}")
        return None
    try:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        data.index.names = ['date']
        logging.info(f"Data loaded successfully from {filename}")
        return data
    except Exception as e:
        logging.error(f"An error occurred while loading data from {filename}: {e}")
        return None

def preprocess_and_save_data(data, scaler_path='scaler.pkl'):
    logging.info("Starting preprocessing data")
    try:
        data = data.drop(columns='Unnamed: 0', errors='ignore').dropna(axis=1, how='all')
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        logging.info("Data preprocessing and scaler saving successful")
        return data
    except Exception as e:
        logging.error(f"Error during preprocessing and saving scaler: {e}")
        return None

def create_sequences(data, sequence_length, num_features):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        if sequence.shape[0] == sequence_length:
            sequences.append(sequence)
    return np.array(sequences)

if __name__ == "__main__":
    stocks = ['SPY', 'META', 'TSLA', 'AMZN', 'MSFT', 'AAPL', 'GOOG', 'NVDA', 'VOO']
    sequence_length = 30  # Adjust this to match the sequence length you're using for training
    num_features = 5  # This should be set based on the actual number of features your model expects

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for stock in stocks:
        logging.info(f"Processing {stock}")
        input_filename = f'p/Users/williampratt/Library/Mobile Documents/com~apple~CloudDocs/Documents/project_sea_ranch/data/raw/{stock}_intraday_1min.csv'
        data = load_data(input_filename)
        if data is None:
            continue
        processed_data = preprocess_and_save_data(data)
        if processed_data is not None:
            sequences = create_sequences(processed_data.values, sequence_length, num_features)
            logging.info(f"Processed data shape for {stock}: {sequences.shape}")

            if sequences.size > 0:
                X = sequences[:, :-1, :]  # All but the last time step
                y = sequences[:, -1, :]   # Just the last time step
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.2)
                X_train_list.append(X_train_split)
                X_test_list.append(X_test_split)
                y_train_list.append(y_train_split)
                y_test_list.append(y_test_split)

    if X_train_list and X_test_list and y_train_list and y_test_list:
        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        for dataset, name in zip([X_train, y_train, X_test, y_test], ['x_train', 'y_train', 'x_test', 'y_test']):
            output_path = r'/Users/williampratt/Library/Mobile Documents/com~apple~CloudDocs/Documents/project_sea_ranch/data/preprocessed data/{}.npy'.format(name)
            np.save(output_path, dataset)
            logging.info(f"Saved {name} data with shape: {dataset.shape}")
    else:
        logging.error("No training or testing data available for any stock.")