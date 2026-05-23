import logging
import pandas as pd
import mlflow
import os
import gc
import json

from src.data_loader import load_data
from src.preprocess import clean_and_preprocess
from src.feature_engineering import add_time_and_promo_features, encode_categorical_features, drop_unused_columns
from train import train_xgboost, train_random_forest
from eval import evaluate_test_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_pipeline():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("Rossmann_Production_Pipeline")

    with mlflow.start_run(run_name="Full_Pipeline_Execution"):
        logging.info("=== Loading & Splitting Data ===")

        df= load_data("data/raw/train.csv", "data/raw/store.csv")
        
        logging.info("=== Preprocessing ===")
        df_prep = clean_and_preprocess(df)
        
        logging.info("=== Feature Engineering ===")
        df_fe = add_time_and_promo_features(df_prep)
        
        logging.info("=== Categorical Encoding ===")
        df_enc, _ = encode_categorical_features(df_fe, df_fe)

        logging.info("=== Dropping Unused Columns ===")
        df_final, _ = drop_unused_columns(df_enc, df_enc)

        # 1. Chronological Split
        # We use the last period of the dataset for testing to simulate a real-world forecasting scenario
        split_date = "2015-06-15"

        # Creating copies to avoid SettingWithCopyWarning
        train_df = df_final[df_final['Date'] <= split_date].copy()
        test_df = df_final[df_final['Date'] > split_date].copy()

        # 2. Separate Features (X) and Target (y)
        # We drop 'Sales' (target) and 'Date' (no longer needed as a feature after splitting)
        X_train_raw = train_df.drop(['Sales', 'Date'], axis=1).astype('float32')
        y_train_raw = train_df['Sales'].values.reshape(-1, 1).astype('float32')

        X_test_raw = test_df.drop(['Sales', 'Date'], axis=1).astype('float32')
        y_test_raw = test_df['Sales'].values.reshape(-1, 1).astype('float32')

        # 3. Memory Cleanup
        # Delete large dataframes that are no longer needed to free up RAM
        del train_df
        del test_df
        gc.collect()

        logging.info(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

        features_list = list(X_train_raw.columns)
        with open("features.json", "w") as f:
            json.dump(features_list, f)
        mlflow.log_artifact("features.json", artifact_path="metadata")

        # ======================= XGBoost Regressor =========================
        xgb_model = train_xgboost(X_train_raw, y_train_raw)
        
        logging.info("=== Testing & Evaluation ===")
        xgb_preds, xgb_rmse, xgb_r2, xgb_plot = evaluate_test_data(X_test_raw, y_test_raw.ravel(), model_type="XGBoost")

        mlflow.log_params({
            "xgb_n_estimators" : 2000,
            "xgb_learning_rate" : 0.02,
            "xgb_max_depth" : 10, 
            "xgb_subsample" : 0.9,
            "xgb_colsample_bytree" : 0.7,
            "xgb_reg_alpha" : 0.1,
            "xgb_reg_lambda" : 1.0,
            "xgb_tree_method" : 'hist',
            "xgb_objective" : "reg:squarederror"
        })

        mlflow.log_metrics({"XGBoost_RMSE": xgb_rmse, "XGBoost_R2_Score": xgb_r2})

        mlflow.log_artifact(xgb_plot, artifact_path="plots")
        if os.path.exists(xgb_plot): os.remove(xgb_plot)

        mlflow.xgboost.log_model(xgb_model, name="xgboost_model")

        # ======================= Random Forest =========================
        rf_model = train_random_forest(X_train_raw, y_train_raw)
        logging.info("=== Testing & Evaluation ===")
        rf_preds, rf_rmse, rf_r2, rf_plot = evaluate_test_data(X_test_raw, y_test_raw, model_type="RandomForest")
        mlflow.log_params({
            "rf_n_estimators" : 2000,
            "rf_min_samples_split" : 2,
            "rf_min_samples_leaf" : 1, 
            "rf_max_features" : 'sqrt',
        })
        mlflow.log_metrics({"rf_RMSE": rf_rmse, "rf_R2_Score": rf_r2})
        mlflow.log_artifact(rf_plot, artifact_path="plots")
        if os.path.exists(rf_plot): os.remove(rf_plot)
        mlflow.sklearn.log_model(rf_model, name="random_forest_model")
        
        logging.info(f"Pipeline executed successfully! First 5 XGBoost Predictions: {xgb_preds[:5]}")

if __name__ == "__main__":
    main_pipeline()