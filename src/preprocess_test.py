"""
Preprocess data:
- Fixing data types
- handeling missing values 
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean data, handle missing values, fixing data types
    """

    logging.info("Starting Preproccessing step")

    # create copy of data
    df_clean = df.copy()

    # Filling missing values in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'] with zero
    df_clean[['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
              'Promo2SinceWeek', 'Promo2SinceYear']] = df_clean[['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                                                                'Promo2SinceWeek', 'Promo2SinceYear']].fillna(0)

    # Filling missing values in CompetitionDistance with mean
    df_clean['CompetitionDistance'] = df_clean['CompetitionDistance'].fillna(df_clean['CompetitionDistance'].mean())

    # Filling missing values in PromoInterval with 'No Promo'
    df_clean['PromoInterval'] = df_clean['PromoInterval'].fillna('No Promo')

    # Fixing values (Convert int values to string)
    df_clean['StateHoliday'] = df_clean['StateHoliday'].replace(0, '0')

    logging.info("Preprocessing complete.")
    return df_clean
