import sys
sys.path.append('../src')  # Ensure access to modules in the src directory

import pandas as pd
from data_preparation import load_data, clean_data, preprocess_data

# Replace 'your_test_data.csv' with the path to an actual test CSV file
test_data_path = 'your_test_data.csv'

# Test load_data function
def test_load_data():
    try:
        data = load_data(test_data_path)
        assert isinstance(data, pd.DataFrame)
        print("load_data function passed.")
    except AssertionError:
        print("load_data function failed.")

# Test clean_data function
def test_clean_data():
    try:
        raw_data = load_data(test_data_path)
        cleaned_data = clean_data(raw_data)
        assert isinstance(cleaned_data, pd.DataFrame)
        print("clean_data function passed.")
    except AssertionError:
        print("clean_data function failed.")

# Test preprocess_data function
def test_preprocess_data():
    try:
        raw_data = load_data(test_data_path)
        cleaned_data = clean_data(raw_data)
        preprocessed_data = preprocess_data(cleaned_data)
        assert isinstance(preprocessed_data, pd.DataFrame)
        print("preprocess_data function passed.")
    except AssertionError:
        print("preprocess_data function failed.")

# Run tests
if __name__ == "__main__":
    test_load_data()
    test_clean_data()
    test_preprocess_data()

