import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from data_processor import get_user_scores

def evaluate_model_on_new_data(username):
    new_data = get_user_scores(username)

    # Load the trained model and scaler
    trained_model = tf.keras.models.load_model("model.keras")
    loaded_scaler = joblib.load("scaler.pkl")

    # Prepare new data
    new_data_df = pd.DataFrame(new_data)  # Assuming `new_data` is a list of dictionaries
    X_new = new_data_df.drop("actualPP", axis=1)
    y_actual = new_data_df["actualPP"]

    # Scale the features
    X_new_scaled = loaded_scaler.transform(X_new)

    # Make predictions
    y_pred = trained_model.predict(X_new_scaled).flatten()

    # Create a DataFrame to sort actual and predicted values together
    results_df = pd.DataFrame({
        "actualPP": y_actual,
        "predictedPP": y_pred
    })

    # Sort by actualPP in descending order
    results_df = results_df.sort_values(by="actualPP", ascending=False).reset_index(drop=True)

    # Plot both actualPP and predictedPP
    plt.plot(results_df["actualPP"], label="Actual PP", marker="o")
    plt.plot(results_df["predictedPP"], label="Predicted PP", marker="x")
    plt.xlabel("Score Rank (Descending by Actual PP)")
    plt.ylabel("PP Value")
    plt.title(f"Actual vs Predicted PP for {username}")
    plt.legend()
    plt.grid()
    plt.show()

    return y_actual, y_pred

# Evaluate the model on new data
actualPP, predictedPP = evaluate_model_on_new_data("[- Yami -]")
