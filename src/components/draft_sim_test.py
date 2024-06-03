# Combined Backtesting Simulator Script

# Imports
import os
import sys
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Configurations
load_dotenv()  # Load environment variables from .env file
logging.basicConfig(level=logging.INFO)

# Global Variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Define your utility functions and classes here

# Data Downloading
def fetch_stock_data(symbol, interval='1min', outputsize='full'):
    def fetch_stock_data(symbol, interval='1min', outputsize='full'):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    # Ensure you're passing the interval and outputsize to the URL
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}&datatype=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        time_series_key = f"Time Series ({interval})"
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            print(f"No data found for {symbol}. Check the API response for error messages.")
            return None
        
        # Parse the JSON data into a DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index').astype(float)
        df.index = pd.to_datetime(df.index)
        df.rename(columns=lambda s: s[3:], inplace=True)  # Removing the numerical prefix from column names.

        # Save the DataFrame as a CSV file.
        df.to_csv(f'/Users/williampratt/Documents/project_sea_ranch/data/raw/{symbol}_intraday_{interval}.csv')
        
        print(f"Data for {symbol} fetched and saved successfully.")
        return df
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        return None
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
        return None
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
        return None
    
# Specify the stock symbols
symbols = ['SPY', 'NVDA', 'VOO']
# Fetch and save data for each stock symbol
for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    stock_data = fetch_stock_data(symbol)
    if stock_data is None:
        print(f"Failed to fetch data for {symbol}.")
    
    # Sleep for 1 minute before the next API call
    print(f"Waiting for 1 minute before fetching the next symbol...")
    time.sleep(60)  # Sleeps for 60 seconds

    pass

# Data Preparation
def load_data(filename):
    # Implementation remains the same
    pass

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
    stocks = ['BLK', 'SPY', 'VOO', 'TSM', 'ASML', 'TCEHY'
        'MSFT',  # Microsoft Corporation
        'AAPL',  # Apple Inc
        'NVDA',  # NVIDIA Corporation
        'AMZN',  # Amazon.com, Inc.
        'GOOG',  # Alphabet Inc.
        'GOOGL',  # Alphabet Inc.
        'META',  # Meta Platforms, Inc.
        'BRK.B',  # Berkshire Hathaway Inc.
        'BRK.A',  # Berkshire Hathaway Inc.
        'LLY',  # Eli Lilly and Company
        'AVGO',  # Broadcom Inc.
        'JPM',  # JPMorgan Chase & Co.
        'V',  # Visa Inc.
        'TSLA',  # Tesla, Inc.
        'WMT',  # Walmart Inc.
        'XOM',  # Exxon Mobil Corporation
        'MA',  # Mastercard Incorporated
        'UNH',  # UnitedHealth Group Incorporated
        'PG',  # The Procter & Gamble Company
        'JNJ',  # Johnson & Johnson
        'HD',  # The Home Depot, Inc.
        'ORCL',  # Oracle Corporation
        'MRK',  # Merck & Co., Inc.
        'COST',  # Costco Wholesale Corporation
        'ABBV',  # AbbVie Inc.
        'CVX',  # Chevron Corporation
        'CRM',  # Salesforce, Inc.
        'BAC',  # Bank of America Corporation
        'AMD',  # Advanced Micro Devices, Inc.
        'NFLX',  # Netflix, Inc.
        'KO',  # The Coca-Cola Company
        'PEP',  # PepsiCo, Inc.
        'LIN',  # Linde plc
        'TMO',  # Thermo Fisher Scientific Inc.
        'ADBE',  # Adobe Inc.
        'DIS',  # The Walt Disney Company
        'ACN',  # Accenture plc
        'WFC',  # Wells Fargo & Company
        'CSCO',  # Cisco Systems, Inc.
        'ABT',  # Abbott Laboratories
        'MCD',  # McDonald's Corporation
        'QCOM',  # QUALCOMM Incorporated
        'TMUS',  # T-Mobile US, Inc.
        'CAT',  # Caterpillar Inc.
        'DHR',  # Danaher Corporation
        'INTU',  # Intuit Inc.
        'VZ',  # Verizon Communications Inc.
        'IBM',  # International Business Machines Corporation
        'AMAT',  # Applied Materials, Inc.
        'GE',  # GE Aerospace
        'INTC']  # Intel Corporation

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

    pass

def create_lstm_model(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    """
    Create a basic LSTM model.
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=units),
        Dropout(dropout_rate),
        Dense(1)  # Single output neuron for a single predicted value
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Train the LSTM model.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained LSTM model.
    """
    performance = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss, Test MAE:", performance)
    return performance

def save_model(model, filename):
    """
    Save the LSTM model to a file.
    """
    model.save(filename)

if __name__ == "__main__":
    # Load the preprocessed data
    X_train = np.load('/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/x_train.npy')
    y_train = np.load('/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/y_train.npy')
    X_test = np.load('/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/x_test.npy')
    y_test = np.load('/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/y_test.npy')

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    if X_train.size == 0 or y_train.size == 0:
        print("Training data is empty. Check data preparation steps.")
    else:
        # Define input shape based on the loaded data
        input_shape = (X_train.shape[1], X_train.shape[2])

    # Create and train the LSTM model
    model = create_lstm_model(input_shape)
    history = train_model(model, X_train, y_train, epochs=10, batch_size=32)

    # Evaluate and save the model
    performance = evaluate_model(model, X_test, y_test)
    save_model(model, '/Users/williampratt/Documents/project_sea_ranch/models/lstm_prediction_model_v1.keras')

# Backtest Strategy
class TradingStrategy:
    def __init__(self, model, alpaca_api_key, alpaca_secret_key, threshold=0.005, capital=10000, stop_loss=0.004, take_profit=0.005):
        self.model = model
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.threshold = threshold
        self.capital = capital
        self.balance = capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.open_positions = {}
        self.stocks = ['BLK', 'SPY', 'VOO', 'TSM', 'ASML', 'TCEHY'
            'MSFT',  # Microsoft Corporation
            'AAPL',  # Apple Inc
            'NVDA',  # NVIDIA Corporation
            'AMZN',  # Amazon.com, Inc.
            'GOOG',  # Alphabet Inc.
            'GOOGL',  # Alphabet Inc.
            'META',  # Meta Platforms, Inc.
            'BRK.B',  # Berkshire Hathaway Inc.
            'BRK.A',  # Berkshire Hathaway Inc.
            'LLY',  # Eli Lilly and Company
            'AVGO',  # Broadcom Inc.
            'JPM',  # JPMorgan Chase & Co.
            'V',  # Visa Inc.
            'TSLA',  # Tesla, Inc.
            'WMT',  # Walmart Inc.
            'XOM',  # Exxon Mobil Corporation
            'MA',  # Mastercard Incorporated
            'UNH',  # UnitedHealth Group Incorporated
            'PG',  # The Procter & Gamble Company
            'JNJ',  # Johnson & Johnson
            'HD',  # The Home Depot, Inc.
            'ORCL',  # Oracle Corporation
            'MRK',  # Merck & Co., Inc.
            'COST',  # Costco Wholesale Corporation
            'ABBV',  # AbbVie Inc.
            'CVX',  # Chevron Corporation
            'CRM',  # Salesforce, Inc.
            'BAC',  # Bank of America Corporation
            'AMD',  # Advanced Micro Devices, Inc.
            'NFLX',  # Netflix, Inc.
            'KO',  # The Coca-Cola Company
            'PEP',  # PepsiCo, Inc.
            'LIN',  # Linde plc
            'TMO',  # Thermo Fisher Scientific Inc.
            'ADBE',  # Adobe Inc.
            'DIS',  # The Walt Disney Company
            'ACN',  # Accenture plc
            'WFC',  # Wells Fargo & Company
            'CSCO',  # Cisco Systems, Inc.
            'ABT',  # Abbott Laboratories
            'MCD',  # McDonald's Corporation
            'QCOM',  # QUALCOMM Incorporated
            'TMUS',  # T-Mobile US, Inc.
            'CAT',  # Caterpillar Inc.
            'DHR',  # Danaher Corporation
            'INTU',  # Intuit Inc.
            'VZ',  # Verizon Communications Inc.
            'IBM',  # International Business Machines Corporation
            'AMAT',  # Applied Materials, Inc.
            'GE',  # GE Aerospace
            'INTC']  # Intel Corporation
        self.trade_history = []
        self.init_alpaca_client()
        self.last_order_time = None

        logging.basicConfig(level=logging.INFO, filename='trading_strategy.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Trading strategy initialized")

        pass

    def preprocess_realtime_data(self, current_data, sequence_length=30):
        if not isinstance(current_data, pd.DataFrame):
            logging.error("current_data is not a DataFrame.")
            return None

        try:
            # Check if date column is set as index
            if 'Unnamed: 0' in current_data.columns and not isinstance(current_data.index, pd.DatetimeIndex):
                current_data.set_index('Unnamed: 0', inplace=True)
                current_data.index = pd.to_datetime(current_data.index)

            # Define the numeric columns for scaling
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']

            # Load the saved scaler
            with open('/Users/williampratt/Documents/project_sea_ranch/scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)

            # Check if the expected columns are in the DataFrame
            if not all(col in current_data.columns for col in numeric_columns):
                logging.error("Some expected columns are missing in current_data.")
                return None

            # Apply the scaler to the numeric columns
            current_data[numeric_columns] = scaler.transform(current_data[numeric_columns])

            # Handle different lengths of data
            if len(current_data) < sequence_length:
                padding = pd.DataFrame(np.zeros((sequence_length - len(current_data), len(numeric_columns))), columns=numeric_columns, index=pd.date_range(start=current_data.index.min(), periods=sequence_length, freq='T')[-sequence_length:])
                current_data = pd.concat([padding, current_data])
            elif len(current_data) > sequence_length:
                current_data = current_data.iloc[-sequence_length:]

            # Check the shape before reshaping
            expected_shape = (sequence_length, len(numeric_columns))
            if current_data[numeric_columns].shape != expected_shape:
                logging.error(f"Data shape after processing does not match expected shape. Got: {current_data[numeric_columns].shape}, expected: {expected_shape}")
                return None

            # Reshape data for LSTM model
            preprocessed_data = current_data[numeric_columns].values.reshape(1, sequence_length, -1)
            return preprocessed_data

        except Exception as e:
            logging.error(f"Error in preprocess_realtime_data: {e}")
            return None

        pass

    def generate_signals(self, prediction, symbol):
        """
        Generate trading signals based on the model prediction for a given symbol.
        """
         # Print statements for debugging
        print(f"generate_signals called with prediction: {prediction}")
        print(f"generate_signals called for symbol: {symbol}")

        signals = []

        # Dynamic threshold calculation
        buy_threshold, sell_threshold = self.calculate_dynamic_thresholds(prediction)

        # Generate signal for the given symbol
        signal = self.interpret_prediction(prediction[0], symbol)
        signals.append(signal)
        logging.info(f"Inside generate_signals function for {symbol}")
        logging.info(f"Prediction received: {prediction[0]}")
        logging.info(f"Signal generated: {signal}")

        return signals
    
        pass

# Main Execution Logic
if __name__ == "__main__":
    symbols = ['BLK', 'SPY', 'VOO', 'TSM', 'ASML', 'TCEHY'
        'MSFT',  # Microsoft Corporation
        'AAPL',  # Apple Inc
        'NVDA',  # NVIDIA Corporation
        'AMZN',  # Amazon.com, Inc.
        'GOOG',  # Alphabet Inc.
        'GOOGL',  # Alphabet Inc.
        'META',  # Meta Platforms, Inc.
        'BRK.B',  # Berkshire Hathaway Inc.
        'BRK.A',  # Berkshire Hathaway Inc.
        'LLY',  # Eli Lilly and Company
        'AVGO',  # Broadcom Inc.
        'JPM',  # JPMorgan Chase & Co.
        'V',  # Visa Inc.
        'TSLA',  # Tesla, Inc.
        'WMT',  # Walmart Inc.
        'XOM',  # Exxon Mobil Corporation
        'MA',  # Mastercard Incorporated
        'UNH',  # UnitedHealth Group Incorporated
        'PG',  # The Procter & Gamble Company
        'JNJ',  # Johnson & Johnson
        'HD',  # The Home Depot, Inc.
        'ORCL',  # Oracle Corporation
        'MRK',  # Merck & Co., Inc.
        'COST',  # Costco Wholesale Corporation
        'ABBV',  # AbbVie Inc.
        'CVX',  # Chevron Corporation
        'CRM',  # Salesforce, Inc.
        'BAC',  # Bank of America Corporation
        'AMD',  # Advanced Micro Devices, Inc.
        'NFLX',  # Netflix, Inc.
        'KO',  # The Coca-Cola Company
        'PEP',  # PepsiCo, Inc.
        'LIN',  # Linde plc
        'TMO',  # Thermo Fisher Scientific Inc.
        'ADBE',  # Adobe Inc.
        'DIS',  # The Walt Disney Company
        'ACN',  # Accenture plc
        'WFC',  # Wells Fargo & Company
        'CSCO',  # Cisco Systems, Inc.
        'ABT',  # Abbott Laboratories
        'MCD',  # McDonald's Corporation
        'QCOM',  # QUALCOMM Incorporated
        'TMUS',  # T-Mobile US, Inc.
        'CAT',  # Caterpillar Inc.
        'DHR',  # Danaher Corporation
        'INTU',  # Intuit Inc.
        'VZ',  # Verizon Communications Inc.
        'IBM',  # International Business Machines Corporation
        'AMAT',  # Applied Materials, Inc.
        'GE',  # GE Aerospace
        'INTC']  # Intel Corporation
    
    # Step 1: Download and save stock data
    for symbol in symbols:
        # Assuming this function saves data and possibly returns a DataFrame
        fetch_stock_data(symbol)

    # Step 2: Data Preparation
    # This might involve loading data, preprocessing it, and then saving the preprocessed data

    # Step 3: Train LSTM Model
    # Load the preprocessed data, create a model, train it, evaluate, and then save the model

    # Step 4: Backtest Strategy
    # Initialize your trading strategy, load the trained model, and run the backtest

    # Note: You'll need to fill in each step with the appropriate calls to the functions
    # and classes defined above, handling data passed between parts as necessary.