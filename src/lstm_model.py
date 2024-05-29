import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import yaml


def create_lstm_model(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    """
    Create a basic LSTM model.
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=units),
        Dropout(dropout_rate),
        Dense(input_shape[1])  # Single output neuron for a single predicted value
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
    with open('src/settings.yml', 'r') as f:
        dat = yaml.load(f, Loader=yaml.SafeLoader)
    # Load the preprocessed data
    X_train = np.load('data/preprocessed data/x_train.npy')
    y_train = np.load('data/preprocessed data/y_train.npy')
    X_test = np.load('data/preprocessed data/x_test.npy')
    y_test = np.load('data/preprocessed data/y_test.npy')

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
    save_model(model, dat['model_path'])