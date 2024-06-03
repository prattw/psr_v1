import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import yaml


logging.basicConfig(level=logging.INFO)

def create_complex_lstm_model(input_shape, units=50, dropout_rate=0.3, learning_rate=0.0005):
    """
    Create a more complex LSTM model with additional layers and more units.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=units, return_sequences=True),  # First LSTM layer
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=True),  # Second LSTM layer with return_sequences=True for stacking
        Dropout(dropout_rate),
        LSTM(units=units),  # Third LSTM layer, now returning only the last output
        Dropout(dropout_rate),
        Dense(input_shape[1])  # Single output neuron for a single predicted value
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
    
    callbacks = [checkpoint, early_stop, reduce_lr]  # Proper list of callbacks

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.1, callbacks=callbacks, verbose=1)  # Use 'callbacks' list here
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
    with open('src/settings.yml', 'r') as f:
        dat = yaml.load(f, Loader=yaml.SafeLoader)
    # Load the preprocessed data

    X_train = np.load('data/preprocessed data/x_train.npy')
    y_train = np.load('data/preprocessed data/y_train.npy')
    X_test = np.load('data/preprocessed data/x_test.npy')
    y_test = np.load('data/preprocessed data/y_test.npy')


    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[1]  # This assumes y_train is 2D with shape [samples, features]

    if X_train.size == 0 or y_train.size == 0:
        print("Training data is empty. Check data preparation steps.")
    else:
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Corrected function name for creating the model
        model = create_complex_lstm_model(input_shape)  # Using the correct function name

        history = train_model(model, X_train, y_train, epochs=50, batch_size=49)


    # Evaluate and save the model
    performance = evaluate_model(model, X_test, y_test)
    save_model(model, dat['model_path'])
