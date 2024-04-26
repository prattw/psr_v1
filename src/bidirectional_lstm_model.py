import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

def create_more_complex_bidirectional_lstm_model(input_shape, units=50, output_units=5, dropout_rate=0.3, learning_rate=0.0005):
    """
    Create a more complex LSTM model with bidirectional LSTM layers and increased units.
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(units=units, return_sequences=True)),  # First layer
        Dropout(dropout_rate),
        Bidirectional(LSTM(units=units, return_sequences=True)),  # Second layer
        Dropout(dropout_rate),
        Bidirectional(LSTM(units=units)),                         # Third layer
        Dropout(dropout_rate),
        Dense(output_units)  # Adjusted to match the number of output features (e.g., 5 if y_train has 5 features)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=49):
    """
    Train the LSTM model.
    """
    checkpoint_path = "models/lstm_best_model.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001, mode='min')
    
    callbacks = [checkpoint, early_stop, reduce_lr]

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.1, callbacks=callbacks, verbose=1)
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained LSTM model.
    """
    performance = model.evaluate(X_test, y_test, verbose=0)
    logging.info("Test Loss, Test MAE: %s", performance)

    return performance

def save_model(model, filename):
    """
    Save the LSTM model to a file.
    """
    model.save(filename)

if __name__ == "__main__":
    # Load the preprocessed data
    data_dir = '/Users/williampratt/Library/Mobile Documents/com~apple~CloudDocs/Documents/project_sea_ranch/data/preprocessed data'
    X_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[1]  # This assumes y_train is 2D with shape [samples, features]

    model = create_more_complex_bidirectional_lstm_model(input_shape, output_units=output_units)
    # Corrected to use the bidirectional LSTM model function

    history = train_model(model, X_train, y_train, epochs=50, batch_size=49)
    performance = evaluate_model(model, X_test, y_test)
    save_model(model, os.path.join(data_dir, '/Users/williampratt/Library/Mobile Documents/com~apple~CloudDocs/Documents/project_sea_ranch/models/lstm_prediction_model_v2.keras'))