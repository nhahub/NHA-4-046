"""
Load data 
"""

import pandas as pd 
import logging

logging.basicConfig(level=logging.INFO)


def load_data(train_path: str, store_path: str):
    """
    Loads train and Store Data, merge them,
    and returns the merged DataFrame.
    """
    logging.info("Loading datasets")

    train_df = pd.read_csv(train_path, parse_dates=['Date'])
    store_df = pd.read_csv(store_path)

    logging.info("Merging datasets")

    # Merging on the common 'store' column
    merged_df = pd.merge(train_df, store_df, on='Store', how='inner')

    logging.info("Sorting data by Date to prevent data leakage")

    # Sorting data by date ascending 
    merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)

    return merged_df