import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
from data_processor import get_data

def train_xgb_model():
    # Removed n_estimators from params
    params = {
        "max_depth": 15,
        "learning_rate": 0.005,
        "random_state": 42,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_jobs": -1
    }

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
        num_boost_round=3000,
        evals=[(dtrain, 'train')],
        evals_result=progress,
        verbose_eval=10  # Print progress every 10 rounds
    )

    # Save the model
    joblib.dump(model, "xgb_model_optimized.pkl.z", compress=('zlib', 3))
    print("\nModel saved to xgb_model_optimized.pkl.z with zlib compression level 3")

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

if __name__ == "__main__":
    print("Starting optimized XGBoost training...")
    model = train_xgb_model()