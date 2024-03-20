# tune_lstm.py
import sys
import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch
from data_preparation import load_data, preprocess_data, create_sequences

# Define the LSTM model builder function
def build_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        return_sequences=True,
        input_shape=input_shape))
    model.add(keras.layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
    model.add(keras.layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error')

    return model

# Load and preprocess your data
# Make sure to replace 'path_to_your_data.csv' with the correct path to your dataset
raw_data = load_data('path_to_your_data.csv')
cleaned_data = preprocess_data(raw_data)
sequences = create_sequences(cleaned_data, sequence_length=20)  # Adjust 'sequence_length' as needed
X_train, y_train = sequences  # Adjust based on how your `create_sequences` function structures the output

# Define input shape based on your dataset
input_shape = X_train.shape[1:]  # Adjust this based on your preprocessed data shape

# Set up the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='tuning',
    project_name='LSTM_tuning')

# Start the hyperparameter search process
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()