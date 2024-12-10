import sys
sys.path.append(f'{sys.path[0]}/..')
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
from data_processor import get_data

parameters_map = {
    "ultra": {
        "max_depth": 13,
        "learning_rate": 0.0085,
        "min_child_weight": 1.1,
        "num_boost_round": 2600
    },
    "plus": {
        "max_depth": 12,
        "learning_rate": 0.01,
        "min_child_weight": 1.3,
        "num_boost_round": 2400
    },
    "standard": {
        "max_depth": 10,
        "learning_rate": 0.013,
        "min_child_weight": 1.7,
        "num_boost_round": 1800
    },
    "lite": {
        "max_depth": 8,
        "learning_rate": 0.018,
        "min_child_weight": 2,
        "num_boost_round": 1200
    },
    "pocket": {
        "max_depth": 7,
        "learning_rate": 0.024,
        "min_child_weight": 2.3,
        "num_boost_round": 800
    }
}

def train_xgb_model(model_version="ultra"):
    params = {
        "max_depth": parameters_map[model_version]["max_depth"],
        "learning_rate": parameters_map[model_version]["learning_rate"],
        "random_state": 42,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "min_child_weight": parameters_map[model_version]["min_child_weight"],
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_jobs": -1
    }
    num_boost_round = parameters_map[model_version]["num_boost_round"]

    print("Loading and preparing data...")
    # Use pandas concat instead of numpy vstack to preserve column names
    all_data = pd.concat(get_data(), ignore_index=True)
    
    # Split features and target while keeping column names
    X_train_full = all_data.drop("actualPP", axis=1)
    y_train_full = all_data["actualPP"]

    print(f"Training on {len(X_train_full)} samples...")
    
    # Convert to DMatrix while properly converting feature names to list
    dtrain = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=list(X_train_full.columns))

    # Train the model with a callback for progress
    print("Training model...")  
    progress = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')],
        evals_result=progress,
        verbose_eval=100  # Print progress every 10 rounds
    )

    # Save the model
    print("Saving model...")
    joblib.dump(model, f"xgb/osu_pp_predictor_model_{model_version}.pkl.z", compress=('zlib', 3))
    print(f"\nModel saved to xgb/osu_pp_predictor_model_{model_version}.pkl.z with zlib compression level 3")

    # Get feature importance with actual feature names
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame(
        {'feature': list(importance.keys()), 
         'importance': list(importance.values())}
    )
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)  # Format numbers nicely
    print(importance_df.head(10).to_string(index=False))

    return model

def train_all_models():
    for model_version in parameters_map.keys():
        train_xgb_model(model_version)

if __name__ == "__main__":
    print("Starting optimized XGBoost training...")
    train_xgb_model("extreme")