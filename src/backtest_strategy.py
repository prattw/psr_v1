import pandas as pd
import numpy as np
import sys
import logging
import time
from tensorflow.keras.models import load_model
from trading_strategy import TradingStrategy
from data_preparation import load_data, preprocess_and_save_data
from lstm_model import create_lstm_model
import os
from dotenv import load_dotenv
import yaml

with open('src/settings.yml', 'r') as f:
    dat = yaml.load(f, Loader=yaml.SafeLoader)

# Load environment variables
load_dotenv()

# Retrieve API keys
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

# Load your trained LSTM model
# Load your trained LSTM model
model_path = dat['model_path']
model = load_model(model_path)

# Initialize the trading strategy
strategy = TradingStrategy(model, alpaca_api_key, alpaca_secret_key)

# Define sequence length and symbols
sequence_length = 30
symbols = dat['stocks']

# Backtest for each symbol
for symbol in symbols:
    try:
        file_path = f"{dat['data_path']}/raw/{symbol}_intraday_1min.csv"
        current_data = pd.read_csv(file_path)

        preprocessed_data = strategy.preprocess_realtime_data(current_data, sequence_length=29)  # Adjusted sequence_length to 29
        print(f"Preprocessed data shape for {symbol}: {preprocessed_data.shape}")  # Debugging statement

        preprocessed_data = strategy.preprocess_realtime_data(current_data, sequence_length=29)
        if preprocessed_data is None:
            logging.error(f"Preprocessing failed for {symbol}, skipping...")
            continue  

        if preprocessed_data.shape != (1, 29, model.input_shape[2]):  # Ensure this matches the model's expected shape
            raise ValueError(f"Preprocessed data for {symbol} is not in the correct shape for the model. Expected shape: (1, 29, {model.input_shape[2]})")

        prediction = model.predict(preprocessed_data)

        print(f"Calling generate_signals with prediction: {prediction}, symbol: {symbol}")  # Debugging statement

        signals = strategy.generate_signals(prediction, symbol)
        logging.info(f"Signal for {symbol}: {signals}")

        # Execute trades based on signals
        # ...

    except FileNotFoundError:
        logging.error(f"File not found for symbol {symbol}: {file_path}")
    except Exception as e:
        logging.error(f"Error processing data for {symbol}: {e}")

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

try:
    # Load historical data for each stock
    spy_data = load_data('/Users/williampratt/Documents/project_sea_ranch/data/raw/SPY_intraday_1min.csv')
    nvda_data = load_data('/Users/williampratt/Documents/project_sea_ranch/data/raw/NVDA_intraday_1min.csv')
    voo_data = load_data('/Users/williampratt/Documents/project_sea_ranch/data/raw/VOO_intraday_1min.csv')

    # Define the input shape for your LSTM model
    sequence_length = 30  # Adjust as needed

    strategy.model = model

    for historical_data, symbol in zip([spy_data, nvda_data, voo_data], ['SPY', 'NVDA', 'VOO']):
        if historical_data is None:
            continue

        logging.info(f"DataFrame structure before preprocessing:\n{historical_data.head()}")

        # Convert historical_data to a numpy array
        historical_array = historical_data.values
        sequences = create_sequences(historical_array, sequence_length)

        if sequences.size == 0:
            logging.error("No sequences created. Check the data preprocessing steps.")
            continue

        # Define the input shape for your LSTM model
        input_shape = (sequence_length, historical_array.shape[1])

        # Load/Create the LSTM model
        model = create_lstm_model(input_shape)
        strategy.model = model

        for sequence in sequences:
            sequence_df = pd.DataFrame(sequence, columns=historical_data.columns)
            print("DataFrame created from sequence:")
            print(sequence_df.head())

            # Preprocess the sequence
            preprocessed_sequence = strategy.preprocess_realtime_data(sequence_df, sequence_length)

            # Reshape the preprocessed data for the model
            sequence_reshaped = preprocessed_sequence.reshape(1, sequence_length, -1)

            # Predict using the model
            prediction = model.predict(sequence_reshaped)

            # Generate signals and execute trades based on the prediction
            signals = strategy.generate_signals(prediction)
            # Ensure that signals include 'qty'
            strategy.execute_trades(signals)
   
            last_day_close_price = sequence_df.iloc[-1]['close']
            last_day_date = sequence_df.index[-1]

            for sig in signals:
                sig['price'] = last_day_close_price
                sig['symbol'] = symbol
                strategy.execute_trades(sig, sig['price'], last_day_date, sig['symbol'])

            # After iterating, you can track performance or do additional tasks
            strategy.track_performance()

    def backtest_strategy():
        # Load data, initialize strategy, etc.

        total_iterations = len(preprocess_and_save_data)
        start_time = time.time()

        # Process one iteration to estimate time per iteration
        process_one_iteration(preprocess_and_save_data[0])

        time_per_iteration = time.time() - start_time
        estimated_total_time = time_per_iteration * total_iterations

        print(f"Estimated Total Time: {estimated_total_time:.2f} seconds")

        for iteration, sequence in enumerate(preprocess_and_save_data, start=1):
            # Your existing code to process each sequence

            # Update and print the countdown timer
            elapsed_time = time.time() - start_time
            time_left = estimated_total_time - elapsed_time
            print(f"Iteration {iteration}/{total_iterations}. Estimated time left: {time_left:.2f} seconds", end='\r')

    # Define your process_one_iteration function here
    def process_one_iteration(sequence):
        # Process a single sequence (similar to what you do in your loop)

    # Run the backtest strategy
        backtest_strategy()        

    # Load/Create the LSTM model
    input_shape = (sequence_length, historical_array.shape[1])  # Example shape, adjust as necessary
    model = create_lstm_model(input_shape)  # Make sure this function returns a compiled model

except FileNotFoundError:
    logging.error("File not found. Please check the file path.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error during backtesting: {e}")
    sys.exit(1)