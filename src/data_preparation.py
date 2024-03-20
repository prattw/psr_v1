import pandas as pd
import numpy as np
import pandas as pd
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)

def load_data(filename):
    if not os.path.exists(filename):
        logging.error(f"File not found: {filename}")
        return None
    try:
        # Read CSV file, assuming that the first column is the date which will be used as the index
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        data.index.names = ['date']  # Rename the index to 'date'

        logging.info(f"Data loaded successfully from {filename}")
        return data

    except Exception as e:
        logging.error(f"An error occurred while loading data from {filename}: {e}")
        return None

from sklearn.preprocessing import MinMaxScaler

def preprocess_and_save_data(data, scaler_path='scaler.pkl'):
    try:
        # Drop 'Unnamed: 0' column if it exists and columns with all NaN values
        data = data.drop(columns='Unnamed: 0', errors='ignore').dropna(axis=1, how='all')

        # Select only numeric columns for scaling
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Save the scaler for later use
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)

        return data

    except Exception as e:
        logging.error(f"Error during preprocessing and saving scaler: {e}")
        return None

def split_and_save_data(filename, output_filename, test_size=0.2):
    data = pd.read_csv(filename)

    # Split data (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    # Save test data to CSV
    test_data.to_csv(output_filename, index=False)
    # Optionally, also save train_data if needed

sequence_length = 60  # Adjust as needed
num_features = 5     # Adjust based on your data

def create_sequences(data, sequence_length, num_features):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        if sequence.shape[0] == sequence_length:
            sequences.append(sequence)
    return np.array(sequences)

def add_rolling_average(data, column_name, window_size):
    """
    Adds a rolling average feature to the dataset.
    """
    if column_name not in data.columns:
        print(f"Column {column_name} not found in data.")
        return data

    try:
        data['rolling_avg'] = data[column_name].rolling(window=window_size).mean()
        return data
    except Exception as e:
        print(f"Error in add_rolling_average: {e}")
        return data

def calculate_sharpe_ratio(data, column_name, risk_free_rate=0.0, window_size=252):
    """
    Calculates the rolling Sharpe Ratio of a financial time series.
    """
    if column_name not in data.columns:
        print(f"Column {column_name} not found in data.")
        return data

    try:
        rolling_mean_return = data[column_name].rolling(window=window_size).mean()
        rolling_std_return = data[column_name].rolling(window=window_size).std()

        data['sharpe_ratio'] = (rolling_mean_return - risk_free_rate) / rolling_std_return * np.sqrt(window_size)
        return data
    except Exception as e:
        print(f"Error in calculate_sharpe_ratio: {e}")
        return data

from sklearn.model_selection import train_test_split

def train_test_split_sequences(sequences, test_size=0.2):
    """
    Splits sequences into training and test sets.

    Parameters:
    sequences (numpy.ndarray): Sequences of data.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    tuple: Training and test data splits.
    """
    num_test_samples = int(test_size * len(sequences))
    train_data = sequences[:-num_test_samples]
    test_data = sequences[-num_test_samples:]
    return train_data, test_data

if __name__ == "__main__":
    stocks = ['SPY', 'NVDA', 'VOO']
    sequence_length = 30  # This should match the sequence length you're using for training
    num_features = 5  # Adjust this based on the actual number of features you have

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for stock in stocks:
        input_filename = f'/Users/williampratt/Documents/project_sea_ranch/data/raw/{stock}_intraday_1min.csv'
        
        # Load and preprocess the data
        data = load_data(input_filename)
        if data is not None:
            processed_data = preprocess_and_save_data(data)
            # Now create the sequences from the processed data
            sequences = create_sequences(processed_data.values, sequence_length, num_features)

            print(f"Processed data shape for {stock}:", sequences.shape)

            # If you have sequences for the stock, then split them into training and testing sets
            if sequences.shape[0] != 0:
                X = sequences[:, :-1, :]  # All but the last time step
                y = sequences[:, -1, :]   # Just the last time step
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.2)

                X_train_list.append(X_train_split)
                X_test_list.append(X_test_split)
                y_train_list.append(y_train_split)
                y_test_list.append(y_test_split)

    # Concatenate the lists into arrays only if they are not empty
    if X_train_list and X_test_list and y_train_list and y_test_list:
        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # Save the concatenated arrays
        for dataset, name in zip([X_train, y_train, X_test, y_test], ['x_train', 'y_train', 'x_test', 'y_test']):
            output_path = f'/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/{name}.npy'
            np.save(output_path, dataset)
            print(f"Saved {name} shape: {dataset.shape}")
    else:
        logging.error("No training or testing data available for any stock.")