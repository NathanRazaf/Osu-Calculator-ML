import tensorflow as tf
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib
from data_processor import get_user_scores

# Evaluate Neural Network Model
def evaluate_neural_network(username, architecture_name):
    """
    Evaluate a neural network model for a specific architecture.
    
    Parameters:
        username (str): Username to fetch scores for.
        architecture_name (str): Name of the architecture to evaluate.
    """
    new_data = get_user_scores(username)

    # Prepare new data
    X_new = new_data.drop("actualPP", axis=1)
    y_actual = new_data["actualPP"]

    # Load the architecture-specific scaler
    scaler_path = f"{architecture_name}_scaler.pkl"
    print(f"Loading scaler from: {scaler_path}")
    loaded_scaler = joblib.load(scaler_path)
    
    # Scale the features
    X_new_scaled = loaded_scaler.transform(X_new)

    # Load the architecture-specific neural network model
    model_path = f"{architecture_name}_model.keras"
    print(f"Loading model from: {model_path}")
    trained_model = tf.keras.models.load_model(model_path)

    # Make predictions
    y_pred = trained_model.predict(X_new_scaled).flatten()

    return y_actual, y_pred

# Evaluate XGBoost Model (unchanged)
def evaluate_xgboost(username):
    """
    Evaluate an XGBoost model.
    
    Parameters:
        username (str): Username to fetch scores for.
    """
    new_data = get_user_scores(username)

    # Prepare new data
    X_new = new_data.drop("actualPP", axis=1)
    y_actual = new_data["actualPP"]

    # Load the trained XGBoost model
    model_path = "xgb_model_optimized.pkl.z"
    print(f"Loading model from: {model_path}")
    trained_model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Prepare data for XGBoost
    dtest = xgb.DMatrix(X_new)

    # Make predictions
    y_pred = trained_model.predict(dtest)

    return y_actual, y_pred

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

def evaluate_all_architectures(username, architectures=['ultra_deep']):
    """
    Evaluate all neural network architectures and find the best one.
    
    Parameters:
        username (str): Username to fetch scores for.
        architectures (list): List of architecture names to evaluate.
    """
    best_mae = math.inf
    best_mse = math.inf
    best_architecture = None
    all_results = {}

    for arch in architectures:
        print(f"\nEvaluating {arch} architecture...")
        y_actual, y_pred = evaluate_neural_network(username, arch)
        mse, mae = evaluate_results(y_actual, y_pred)
        all_results[arch] = {
            'mse': mse,
            'mae': mae,
            'y_actual': y_actual,
            'y_pred': y_pred
        }
        
        if mse < best_mse:
            best_mse = mse
            best_mae = mae
            best_architecture = arch

    # Print comparison of all architectures
    print("\nArchitecture Comparison:")
    for arch, results in all_results.items():
        print(f"\n{arch}:")
        print(f"MSE: {results['mse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")

    print(f"\nBest Architecture: {best_architecture}")
    print(f"Best MSE: {best_mse:.4f}")
    print(f"Best MAE: {best_mae:.4f}")

    # Plot results for the best architecture
    plot_and_evaluate_results(
        all_results[best_architecture]['y_actual'],
        all_results[best_architecture]['y_pred'],
        username,
        f"Best Architecture ({best_architecture})"
    )

    return best_architecture, best_mse, best_mae, all_results

# Neural network usage:
# best_arch, mse, mae, results = evaluate_all_architectures("NathanRazaf")

# XGBoost example usage:
y_actual, y_pred = evaluate_xgboost("NathanRazaf")
mse, mae = evaluate_results(y_actual, y_pred)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
plot_and_evaluate_results(y_actual, y_pred, "xotixx", "XGBoost")