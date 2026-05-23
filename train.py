import logging
import json
import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def load_best_params(model_name: str) -> dict:
#     with open("config/best_params.json", "r") as f:
#         params = json.load(f)
#     return params.get(model_name, {})

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    logging.info("Training Final XGBoost Model")
    # params = load_best_params("XGBoost")
    
    model = xgb.XGBRegressor(n_estimators=2000,         
                        learning_rate=0.02,        
                        max_depth=10,              
                        subsample=0.9,             
                        colsample_bytree=0.7,      
                        reg_alpha=0.1,             # L1 regularization 
                        reg_lambda=1.0,            # L2 regularization 
                        random_state=42,
                        n_jobs=-1,                 
                        tree_method='hist',        
                        objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    model.save_model("models/final_xgboost.json")
    logging.info("XGBoost Model saved to models/final_xgboost.json")
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    logging.info("Training Final RandomForest Model")
    # params = load_best_params("RandomForest")
    
    model = RandomForestRegressor(n_estimators=100, 
                                  min_samples_split=2, 
                                  min_samples_leaf=1, 
                                  max_features='sqrt', 
                                  random_state=42, 
                                  n_jobs=-1)
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    with open("models/final_randomforest.pkl", "wb") as f:
        pickle.dump(model, f)
    logging.info("RandomForest Model saved to models/final_randomforest.pkl")
    return model