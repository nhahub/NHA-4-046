import logging
import pickle
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_test_data(X_test: pd.DataFrame, y_test, model_type="XGBoost"):
    logging.info(f"Loading saved {model_type} model for testing")
    
    if model_type == "XGBoost":
        if not os.path.exists("models/final_xgboost.json"):
            raise FileNotFoundError("XGBoost model file not found!")
        model = xgb.XGBRegressor()
        model.load_model("models/final_xgboost.json")
    elif model_type == "RandomForest":
        if not os.path.exists("models/final_randomforest.pkl"):
            raise FileNotFoundError("RandomForest model file not found!")
        with open("models/final_randomforest.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        raise ValueError("Unknown model type!")

    logging.info("Making predictions on test data")
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    logging.info(f"{model_type} - RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")

    # Plotting the difference between y_actual and y_pred
    plt.figure(figsize=(10, 6)) 
    plt.scatter(y_test[:200], predictions[:200], alpha=0.6, color='blue', label='Predictions')
    
    max_val = max(max(y_test[:200]), max(predictions[:200]))
    min_val = min(min(y_test[:200]), min(predictions[:200]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y=x)')
    
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'{model_type} - Actual vs Predicted Sales')
    plt.legend()
    plt.grid(True)

    plot_filename = f"{model_type}_actual_vs_pred.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

    return predictions, rmse, r2, plot_filename