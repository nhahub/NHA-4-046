import logging
import pandas as pd
import numpy as np
import mlflow
import os
import xgboost as xgb

from src.preprocess_test import clean_and_preprocess
from src.feature_engineering import add_time_and_promo_features, encode_categorical_features, drop_unused_columns

def generate_predictions():
    logging.basicConfig(level=logging.INFO)
    logging.info("=== Loading Test & Store Data ===")

    test_df = pd.read_csv("data/raw/test.csv", parse_dates=['Date'])
    store_df = pd.read_csv("data/raw/store.csv")

    df_test_merged = pd.merge(test_df, store_df, on='Store', how='inner')

    test_ids = df_test_merged['Id']

    logging.info("=== Preprocessing Test Data ===")
    df_test_merged['Open'] = df_test_merged['Open'].fillna(1)
    df_prep = clean_and_preprocess(df_test_merged)
    df_fe = add_time_and_promo_features(df_prep)
    df_enc, _ = encode_categorical_features(df_fe, df_fe)
    df_final, _ = drop_unused_columns(df_enc, df_enc)

    X_test_final = df_final.drop(['Id', 'Date'], axis=1, errors='ignore').astype('float32')

    logging.info("=== Loading Model ===")
    model_path = "models/final_xgboost.json"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found in the current directory. Please make sure to save it first.")
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    expected_features = model.get_booster().feature_names
    
    logging.info("=== Aligning Features with Training Data ===")
    for col in expected_features:
        if col not in X_test_final.columns:
            X_test_final[col] = 0
            
    X_test_final = X_test_final[expected_features]
    X_test_final = X_test_final.astype('float32')
    
    logging.info("=== Making Predictions ===")
    preds = model.predict(X_test_final)
    preds = np.where(df_test_merged['Open'] == 0, 0, preds)

    submission = pd.DataFrame({
        "Id": test_ids,
        "Sales": preds
    })

    submission['Sales'] = submission['Sales'].clip(lower=0)

    output_path = "data/processed/submission.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    logging.info("Done Everything is Ok. Predictions made successfully")

if __name__ == "__main__":
    generate_predictions()