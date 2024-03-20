import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
import numpy as np

# Load your trained model
model = load_model('/Users/williampratt/Documents/project_sea_ranch/models/lstm_prediction_model_v1.keras')

# Define the required variables
timesteps = 29  # Number of time steps your model expects
features = 5    # Number of features your model expects

# Define function to prepare the data
def prepare_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    
    # Apply necessary preprocessing steps here
    # Example: Scaling, reshaping to match the model's input shape
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])
    
    # Reshaping to the format expected by the LSTM model
    reshaped_data = np.array([scaled_data[i:i + timesteps] for i in range(len(scaled_data) - timesteps)])
    
    return reshaped_data

# Load and preprocess the test data
nvda_test_data = prepare_data('/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/NVDA_test_data.csv')
spy_test_data = prepare_data('/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/SPY_test_data.csv')
voo_test_data = prepare_data('/Users/williampratt/Documents/project_sea_ranch/data/preprocessed data/VOO_test_data.csv')

# Generate predictions
nvda_predictions = model.predict(nvda_test_data)
spy_predictions = model.predict(spy_test_data)
voo_predictions = model.predict(voo_test_data)

# Function to convert predictions to 'buy'/'sell' signals
def convert_predictions_to_signals(predictions, threshold=0.5):
    # Assuming your model's output is a continuous value,
    # this function will convert it into 'buy' or 'sell' signals based on a threshold.
    signals = ['buy' if pred[0] > threshold else 'sell' for pred in predictions]
    return signals

# Now use the function with the defined threshold
nvda_signals = convert_predictions_to_signals(nvda_predictions, threshold=0.5)
spy_signals = convert_predictions_to_signals(spy_predictions, threshold=0.5)
voo_signals = convert_predictions_to_signals(voo_predictions, threshold=0.5)

def plot_sma_with_predictions(stock_data, predictions, sma_window=60):
    # Calculate the Simple Moving Average
    stock_data['SMA'] = stock_data['close'].rolling(window=sma_window).mean()

    # Plot closing price and SMA
    plt.figure(figsize=(15, 8))
    plt.plot(stock_data['close'], label='Closing Price', color='blue')
    plt.plot(stock_data['SMA'], label=f'{sma_window}-Minute SMA', color='orange')

    # Overlay buy/sell predictions
    for i in range(len(predictions)):
        if predictions[i] == 'buy':
            plt.scatter(stock_data.index[i], stock_data['close'][i], marker='^', color='green')
        elif predictions[i] == 'sell':
            plt.scatter(stock_data.index[i], stock_data['close'][i], marker='v', color='red')

    plt.title('Stock Price with SMA and Model Predictions')
    plt.xlabel('DateTime')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
# Load actual stock data for plotting
nvda_data = pd.read_csv('/Users/williampratt/Documents/project_sea_ranch/data/raw/NVDA_intraday_1min.csv')
spy_data = pd.read_csv('/Users/williampratt/Documents/project_sea_ranch/data/raw/SPY_intraday_1min.csv')
voo_data = pd.read_csv('/Users/williampratt/Documents/project_sea_ranch/data/raw/VOO_intraday_1min.csv')

# Plot for each stock
plot_sma_with_predictions(nvda_data, nvda_signals, sma_window=60)
plot_sma_with_predictions(spy_data, spy_signals, sma_window=60)
plot_sma_with_predictions(voo_data, voo_signals, sma_window=60)