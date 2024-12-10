import sys
sys.path.append(f'{sys.path[0]}/..')
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from data_processor import get_user_scores

_model_cache = {}


def load_model(model_version):
    """Cache and load XGBoost model."""
    if model_version not in _model_cache:
        model_path = f"xgb/osu_pp_predictor_model_{model_version}.pkl.z"
        print(f"Loading model from: {model_path}")
        _model_cache[model_version] = joblib.load(model_path)
    return _model_cache[model_version]

# Evaluate XGBoost Model (unchanged)
def evaluate_xgboost(username, model_version="ultra"):
    """
    Evaluate an XGBoost model.
    
    Parameters:
        username (str): Username to fetch scores for.
    """
    new_data = get_user_scores(username)

    # Prepare new data
    X_new = new_data.drop("actualPP", axis=1)
    y_actual = new_data["actualPP"]

    # Load and evaluate the trained XGBoost model
    trained_model = load_model(model_version)
    dtest = xgb.DMatrix(X_new)
    y_pred = trained_model.predict(dtest)

    return y_actual, y_pred


def evaluate_all_xgboost_models(username):
    """
    Evaluate all XGBoost models and find the best one.
    
    Parameters:
        username (str): Username to fetch scores for.
    """
    maes = {}
    mses = {}

    for model_version in ["ultra", "plus", "standard", "lite"]:
        print(f"\nEvaluating model version {model_version}...")
        y_actual, y_pred = evaluate_xgboost(username, model_version)
        mse, mae = evaluate_results(y_actual, y_pred)
        mses[model_version] = mse
        maes[model_version] = mae

    return maes, mses


def evaluate_models_multiple_users(usernames):
    """
    Evaluate all XGBoost models across multiple users and average the results.
    
    Parameters:
        usernames (list): List of usernames to evaluate models on.
    
    Returns:
        tuple: (average_maes, average_mses) dictionaries containing averaged metrics for each model version
    """
    # Initialize dictionaries to store cumulative errors
    cumulative_maes = {"ultra": 0, "plus": 0, "standard": 0, "lite": 0}
    cumulative_mses = {"ultra": 0, "plus": 0, "standard": 0, "lite": 0}
    
    # Count successful evaluations for each user
    successful_evals = 0
    
    for username in usernames:
        try:
            print(f"\nEvaluating models for user: {username}")
            maes, mses = evaluate_all_xgboost_models(username)
            
            # Add to cumulative totals
            for model_version in cumulative_maes.keys():
                cumulative_maes[model_version] += maes[model_version]
                cumulative_mses[model_version] += mses[model_version]
            
            successful_evals += 1
            
        except Exception as e:
            print(f"Error evaluating user {username}: {str(e)}")
            continue
    
    # Calculate averages
    if successful_evals > 0:
        average_maes = {model: total / successful_evals 
                       for model, total in cumulative_maes.items()}
        average_mses = {model: total / successful_evals 
                       for model, total in cumulative_mses.items()}
    else:
        raise ValueError("No successful evaluations were completed")
    
    print(f"\nSuccessfully evaluated {successful_evals} users")
    return average_maes, average_mses


def plot_xgboost_results(label, maes, mses):
    """
    Plot MAE and MSE results side by side for each XGBoost model version.
    
    Parameters:
        username (str): Username to fetch scores for.
        maes (dict): Dictionary of MAE values for each model version.
        mses (dict): Dictionary of MSE values for each model version.
    """
    plt.figure(figsize=(12, 6))
    
    # Set the positions for the bars
    x = np.arange(len(maes))
    width = 0.35  # Width of the bars
    
    # Create bars
    plt.bar(x - width/2, list(maes.values()), width, label='MAE', color='skyblue')
    plt.bar(x + width/2, list(mses.values()), width, label='MSE', color='lightcoral')
    
    # Customize the plot
    plt.xlabel("Model Version")
    plt.ylabel("Error Value")
    plt.title(f"MAE and MSE Comparison for XGBoost Models ({label})")
    plt.xticks(x, maes.keys())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of each bar
    for i, v in enumerate(maes.values()):
        plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(mses.values()):
        plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def evaluate_results(y_actual, y_pred):
    mse = ((y_actual - y_pred) ** 2).mean()
    mae = (y_actual - y_pred).abs().mean()
    return mse, mae


def plot_and_evaluate_results(y_actual, y_pred, username, model_name):
    """
    Helper function to plot results and calculate evaluation metrics.
    """
    # Create a DataFrame for results
    results_df = pd.DataFrame({
        "actualPP": y_actual,
        "predictedPP": y_pred
    })

    # Calculate evaluation metrics
    mse, mae = evaluate_results(y_actual, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Sort by actualPP in descending order
    results_df = results_df.sort_values(by="actualPP", ascending=False).reset_index(drop=True)

    # Plot both actualPP and predictedPP
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["actualPP"], label="Actual PP", marker="o")
    plt.plot(results_df["predictedPP"], label="Predicted PP", marker="x")
    plt.xlabel("Score Rank (Descending by Actual PP)")
    plt.ylabel("PP Value")
    plt.title(f"Actual vs Predicted PP for {username} ({model_name})")
    plt.legend()
    plt.grid()
    plt.show()


def clear_model_cache():
    """Clear cached models to free memory."""
    global _model_cache
    _model_cache.clear()

# XGBoost example usage:
usernames = ["NathanRazaf", "peppy", "mrekk", "ROGVE ONE", "RayZero", "bored yes"]
# maes, mses = evaluate_models_multiple_users(usernames)
# plot_xgboost_results("6 random users", maes, mses)
# clear_model_cache()

y_actual, y_pred = evaluate_xgboost("peppy", "extreme")
plot_and_evaluate_results(y_actual, y_pred, "RayZero", "lite")