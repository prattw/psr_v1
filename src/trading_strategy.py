import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv
import requests
from datetime import datetime
from datetime import time as dt_time  # Rename to avoid conflict with the 'time' module
import pytz
import time  # Keep this as is for time.sleep functionality
import signal
import sys
import csv
import pickle
import asyncio
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preparation import load_data, preprocess_and_save_data
from lstm_model import create_lstm_model
from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream

load_dotenv()
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

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
        self.stocks = ['SPY', 'NVDA', 'VOO']
        self.trade_history = []
        self.init_alpaca_client()
        self.last_order_time = None

        logging.basicConfig(level=logging.INFO, filename='trading_strategy.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Trading strategy initialized")

    def init_alpaca_client(self):
        try:
            self.api = REST(self.alpaca_api_key, self.alpaca_secret_key, base_url='https://paper-api.alpaca.markets')
            logging.info("Alpaca client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Alpaca REST client: {e}")
            sys.exit(1)

        # Test getting market data
        try:
            symbols = ['SPY', 'NVDA', 'VOO']
            for symbol in symbols:
                # Use get_latest_trade instead of get_last_trade
                last_trade = self.api.get_latest_trade(symbol)
                print(f"Last trade for {symbol}: {last_trade}")
        except Exception as e:
            print(f"An error occurred: {e}")

        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise Exception("Alpaca API key or secret key not found")

        self.market_open = datetime.strptime("09:30:00", "%H:%M:%S").time()
        self.market_close = datetime.strptime("17:30:00", "%H:%M:%S").time()
        self.est = pytz.timezone('US/Eastern')

    async def setup_websocket_connection(self):
        self.stream = Stream(self.alpaca_api_key,
                             self.alpaca_secret_key,
                             base_url='https://paper-api.alpaca.markets',
                             data_feed='iex')  # or use 'sip' for paid subscription

        # Subscribe to trade updates for each stock in your portfolio
        for stock in self.stocks:
            self.stream.subscribe_trades(self.handle_trade_update, stock)

        # Start the stream (no need to run it in a separate thread)
        await self.stream._run_forever()

    async def handle_trade_update(self, trade):
        logging.info(f"Received trade update: {trade}")

    def should_trade_today(self):
        today = datetime.now(self.est).date()
        return today.weekday() < 5  # Monday-Friday

    def is_market_open(self):
        current_time = datetime.now(self.est).time()
        return self.market_open <= current_time <= self.market_close

    def place_time_specific_order(self):
        est = pytz.timezone('US/Eastern')
        current_time_est = datetime.now(est)
        target_time = dt_time(16, 9, 11)  # Target time

        # Check if we have already placed an order this minute
        if self.last_order_time and self.last_order_time.minute == current_time_est.minute:
            return  # Skip if an order has already been placed in the current minute

        if current_time_est.time().hour == target_time.hour and current_time_est.time().minute == target_time.minute:
            print("Placing buy order for NVDA")
            self.place_order('buy', 'NVDA', 11)
            self.last_order_time = current_time_est 

    def process_stock_data(self, time_series):
        if not time_series:
            logging.error("Empty time series data received")
            return None
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = df.rename(columns=lambda s: s.split('. ')[1])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.index = pd.to_datetime(df.index)
        return df

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

    def calculate_dynamic_thresholds(self, predictions):
        mean_prediction = predictions.mean()
        buy_threshold = mean_prediction * 1.00007  # 0.007% above the mean prediction
        sell_threshold = mean_prediction * 0.99998 # 0.002% below the mean prediction
        return buy_threshold, sell_threshold

    def create_signals_based_on_predictions(self, predictions, buy_threshold, sell_threshold):
        signals = []
        for prediction in predictions:
            if prediction > buy_threshold:
                signals.append('buy')
            elif prediction < sell_threshold:
                signals.append('sell')
            else:
                signals.append('hold')
        return signals

    def verify_input_format(model_input, expected_shape):
        """
        Verify if the model input matches the expected shape.

        Args:
        model_input (np.array): The input data for the model.
        expected_shape (tuple): The expected shape of the input data.

        Returns:
        bool: True if the shape matches, False otherwise.
        """
        if model_input.shape != expected_shape:
            logging.error(f"Input shape mismatch. Expected: {expected_shape}, Received: {model_input.shape}")
            return False
        return True

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

    def interpret_prediction(self, prediction, symbol):
        # Interpret the model's output to decide the action
        if prediction > self.threshold:
            return {'action': 'buy', 'symbol': symbol}
        elif prediction < -self.threshold:
            return {'action': 'sell', 'symbol': symbol}
        else:
            return {'action': 'hold', 'symbol': symbol}

    def execute_trades(self, signals):
        """
        Execute trades based on the signals generated by the model.
        """
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            qty = signal['qty']  # Ensure quantity is included in your signal
            logging.info(f"Executing trade for {signal['symbol']}, Action: {signal['action']}")

            if action == 'buy':
                self.place_order('buy', symbol, qty)
                logging.info(f"Placing buy order for {symbol}, qty: {qty}")

            elif action == 'sell':
                self.place_order('sell', symbol, qty)
                logging.info(f"Placing sell order for {symbol}, qty: {qty}")

        pass

    def handle_sell_signal(self, signal):
        """
        Handle a sell signal for a given stock.
        """
        symbol = signal['symbol']
        position_to_sell = self.open_positions.get(symbol)

        if position_to_sell:
            current_price = self.get_current_market_data(symbol)
            purchase_price = position_to_sell['purchase_price']
            time_held = datetime.now(self.est) - position_to_sell['purchase_time']

            # Check if conditions for selling are met
            if (current_price >= purchase_price * (1 + self.take_profit) or
                current_price <= purchase_price * (1 - self.stop_loss) or
                (datetime.now(self.est).time() >= self.market_close and time_held.days < 1)):
                self.place_order('sell', symbol, position_to_sell['qty'])
                self.open_positions.pop(symbol, None)
                logging.info(f"Sold {symbol} due to sell signal or market close.")

    def place_order(self, order_type, symbol, qty):
        logging.info(f"Attempting to place a {order_type} order for {symbol}, qty: {qty}")
        try:
            order_params = {
                "symbol": symbol, 
                "qty": qty, 
                "type": "market", 
                "side": order_type,
                "time_in_force": "day"
            }
            api_url = "https://paper-api.alpaca.markets/v2/orders"
            headers = {
                "APCA-API-KEY-ID": self.alpaca_api_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret_key
            }

            response = requests.post(api_url, json=order_params, headers=headers)
            response.raise_for_status()
            order_response = response.json()

            # Extract price from the response, use a placeholder if not available
            executed_price = order_response.get('filled_avg_price', 'N/A')

            logging.info(f"Order {order_type} placed successfully for {symbol}, qty: {qty}")
            self.last_order_time = datetime.now(pytz.timezone('US/Eastern'))  # Update last order time on success

        except Exception as e:
            logging.error(f"General error in place_order for {symbol}: {e}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception in place_order for {symbol}: {e}")
        except Exception as e:
            logging.error(f"General error in place_order for {symbol}: {e}")

    def get_current_market_data(self, symbol):
        try:
            current_price = self.api.get_latest_trade(symbol).price
            return current_price
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def manage_risk(self):
        for position in self.open_positions:
            current_price = self.get_current_market_data(position['symbol'])
            if current_price <= position['purchase_price'] * (1 - self.stop_loss):
                self.place_order('sell', position['symbol'], position['qty'])  # Replace execute_trade with place_order
                logging.info(f"Stop-loss triggered for {position['symbol']}")
            elif current_price >= position['purchase_price'] * (1 + self.take_profit):
                self.place_order('sell', position['symbol'], position['qty'])  # Replace execute_trade with place_order
                logging.info(f"Take-profit triggered for {position['symbol']}")

    def track_performance(self):
        # Assuming self.trade_history is a list of dictionaries, 
        # each containing details about individual trades

        total_profit = 0
        wins = 0
        losses = 0
        drawdown = 0
        max_drawdown = 0
        peak_balance = self.capital

        for trade in self.trade_history:
            profit = trade['profit']  # Assuming each trade records profit
            total_profit += profit

            if profit > 0:
                wins += 1
            else:
                losses += 1

            # Calculate drawdown
            self.balance += profit
            if self.balance > peak_balance:
                peak_balance = self.balance
            drawdown = peak_balance - self.balance
            max_drawdown = max(max_drawdown, drawdown)

        win_loss_ratio = wins / max(1, losses)
        # sharpe_ratio = self.calculate_sharpe_ratio()  # Implement this method based on your risk-free rate and returns

        # Record the metrics
        performance_metrics = {
            'total_profit': total_profit,
            'win_loss_ratio': win_loss_ratio,
            'max_drawdown': max_drawdown,
            # 'sharpe_ratio': sharpe_ratio
        }

        # You might want to log these metrics or save them to a file
        logging.info(f"Performance Metrics: {performance_metrics}")
        return performance_metrics

        pass

    def backtest(self, historical_data):
        """
        Backtest the trading strategy using historical data.

        Parameters:
        historical_data (DataFrame): Preprocessed historical market data.

        Returns:
        None
        """
        for index, row in historical_data.iterrows():
            # Simulate generating signals for each day
            signal = self.generate_signals(row)

            # Simulate executing trades based on the signal
            self.execute_trades([signal])

            # Manage risk and update open positions
            self.manage_risk()

        # After backtesting, track the overall performance
        self.track_performance()

    

    def signal_handler(self, signum, frame):
        logging.info("Graceful shutdown initiated")
        # Perform any clean-up here
        sys.exit(0)

    def init_signal_handler(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

async def main():
    # Load environment variables
    load_dotenv()

    # Retrieve API keys
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

    # Load your trained LSTM model
    model_path = '/Users/williampratt/Documents/project_sea_ranch/models/lstm_prediction_model_v1.keras'
    model = load_model(model_path)

    # Initialize trading strategy
    trading_strategy = TradingStrategy(model, alpaca_api_key, alpaca_secret_key)
    trading_strategy.init_signal_handler()

    # Start WebSocket connection
    asyncio.create_task(trading_strategy.setup_websocket_connection())

    # Main execution loop
    while True:
        trading_strategy.place_time_specific_order()
        await asyncio.sleep(27)

    # Main execution logic
if __name__ == "__main__":
    asyncio.run(main())