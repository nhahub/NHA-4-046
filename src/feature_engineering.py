"""
Add Some features and encode it 
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO)

def add_time_and_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds basic calendar elements, seasons, and engineering metrics 
    for how long competitions and Promo2 have been open.
    """
    logging.info("Engineering calendar and timeline features")
    fe_df = df.copy()

    # Extract time features
    fe_df['Year'] = fe_df['Date'].dt.year
    fe_df['Month'] = fe_df['Date'].dt.month
    fe_df['Day'] = fe_df['Date'].dt.day
    fe_df['WeekofYear'] = fe_df['Date'].dt.isocalendar().week    

    # add seasons feature
    fe_df['Seasons'] = fe_df['Month'].apply(lambda month: 'Winter' if month in (12, 1, 2) \
                                 else('Spring' if month in (3, 4, 5) \
                                     else('Summer' if month in (6, 7, 8) \
                                         else 'Autumn')))
    
    # Create new column for Competitors Open (in months)
    fe_df['CompetitionOpen_month'] = 12 * (fe_df['Year'] - fe_df['CompetitionOpenSinceYear']) + (fe_df['Month'] - fe_df['CompetitionOpenSinceMonth'])

    # Create new column showing the duration of the promotion in months
    fe_df['Promo2Open'] = 12 * (fe_df['Year'] - fe_df['Promo2SinceYear']) + (fe_df['WeekofYear'] - fe_df['Promo2SinceWeek']) / 4

    return fe_df


def encode_categorical_features(train: pd.DataFrame, eval: pd.DataFrame):
    """
    Transforms string/object categorical features into integers.
    """
    logging.info("Starting Categorical")

    train_df = train.copy()
    test_df = eval.copy()

    nominal_cat =  ['StoreType', 'StateHoliday']
    
    # Encoding categorical data to numerical 
    assortment_order_map = {'a': 0, 'c': 1, 'b': 2}
    train_df['Assortment'] = train_df['Assortment'].replace(assortment_order_map)
    test_df['Assortment'] = test_df['Assortment'].replace(assortment_order_map)

    # Encoding categorical data to numerical 
    seasons_map = {'Winter':0, 'Spring':1, 'Summer': 2, 'Autumn':3}
    train_df['Seasons'] = train_df['Seasons'].replace(seasons_map)
    test_df['Seasons'] = test_df['Seasons'].replace(seasons_map)

    # Encoding Train Data
    one_h_enc_train_model = OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit(train_df[nominal_cat])
    nominal_train_cat = one_h_enc_train_model.transform(train_df[nominal_cat])
    new_cols_name_train = one_h_enc_train_model.get_feature_names_out(nominal_cat)
    encoded_train_df = pd.DataFrame(nominal_train_cat, columns=new_cols_name_train, index=train_df.index)
    train_df = pd.concat([train_df.drop(nominal_cat, axis=1), encoded_train_df], axis=1)

    # Encoding test Data
    nominal_test_cat = one_h_enc_train_model.transform(test_df[nominal_cat])
    new_cols_name_test = one_h_enc_train_model.get_feature_names_out(nominal_cat)
    encoded_test_df = pd.DataFrame(nominal_test_cat, columns=new_cols_name_test, index=test_df.index)
    test_df = pd.concat([test_df.drop(nominal_cat, axis=1), encoded_test_df], axis=1)

    logging.info("Categorical Encoding finished successfully.")
    return train_df, test_df


def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    logging.info("Dropping unused and leaky columns")

    train_df = train.copy()
    eval_df = eval.copy()

    drop_cols = ['PromoInterval', 'Customers']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    eval_df = eval_df.drop(columns=[c for c in drop_cols if c in eval_df.columns], errors="ignore")
    
    logging.info(f"Columns remaining in train: {list(train_df.columns)}")
    return train_df, eval_df