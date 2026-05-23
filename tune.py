import logging
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from src.data_loader import load_data
from src.preprocess import clean_and_preprocess
from src.feature_engineering import add_time_and_promo_features, encode_categorical_features, drop_unused_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_gridsearch():
    logging.info("Starting Hyperparameter Tuning using RandomizedSearch")

    # load and split data
    df= load_data("data/raw/train.csv", "data/raw/store.csv")

    df = df.sort_values('Date').reset_index(drop=True)
    
    cleaned_df = clean_and_preprocess(df)
    featured_df = add_time_and_promo_features(cleaned_df)
    
    # Encode Categorical features in numerical
    X_encoded, _ = encode_categorical_features(featured_df, featured_df)

    # drop unused cols
    df_final, _ = drop_unused_columns(X_encoded, X_encoded)

    # Specify Features (Input Values) and Target (Output values)
    X = df_final.drop(['Sales', 'Date'], axis=1)
    y = df_final['Sales']

    configs = {
        "XGBoost": {
            "model": xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
            "params": {
                "booster" : ["gbtree"],
                "n_estimators" : [100, 500, 1000, 2000],
                "learning_rate" : [0.1, 0.02,0.05],
                "max_depth": [6, 8, 10],
                "subsample": [0.8, 0.9],
                "colsample_bytree" : [0.5, 0.7],
                "reg_alpha" : [0.1],
                "reg_lambda" : [1.0]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 200],              
                "max_depth": [15, 30, None],             
                "min_samples_split": [5],                
                "min_samples_leaf": [2, 4],             
                "max_features": ["sqrt", 1.0]
            }
        }
    }

    best_params_dict = {}

    for model_name, config in configs.items():
        logging.info(f"Running RandomizedSearchCV for {model_name}")
        grid = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=15,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42
        )
        grid.fit(X, y)
        
        logging.info(f"Best {model_name} Params: {grid.best_params_}")
        best_params_dict[model_name] = grid.best_params_
    
    os.makedirs("config", exist_ok=True)
    with open("config/best_params.json", "w") as f:
        json.dump(best_params_dict, f, indent=4)
    logging.info("Tuning complete. Best parameters saved to config/best_params.json")


if __name__ == "__main__":
    run_gridsearch()
