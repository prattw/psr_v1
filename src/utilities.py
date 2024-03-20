import pandas as pd
import logging

# Example utility function for saving data
def save_data_to_csv(data, filename):
    try:
        data.to_csv(filename)
        logging.info(f"Data saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")

# More utility functions...
# Come back to this file, it is not complete!!!!