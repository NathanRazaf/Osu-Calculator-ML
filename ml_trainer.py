import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from data_processor import get_data

# Initialize the StandardScaler
scaler = StandardScaler()

# Build the Neural Network
def create_model():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(15,)),  # Input layer (15 features)
        Dropout(0.2),  # Regularization
        Dense(64, activation="relu"),  # Hidden layer
        Dropout(0.2),
        Dense(32, activation="relu"),  # Hidden layer
        Dropout(0.2),
        Dense(1)  # Output layer (for regression)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

# Train the Model
def train_model():
    model = create_model()
    batch_losses = []  # Store loss for each batch
    
    for batch_index, batch_df in enumerate(get_data()):
        # Preprocess the data
        X_train, y_train = preprocess_data(batch_df, train=True)

        # Train the model on the batch
        history = model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

        # Get the loss for the last epoch and log it
        final_epoch_loss = history.history["loss"][-1]
        batch_losses.append(final_epoch_loss)
        print(f"Batch {batch_index + 1} - Loss: {final_epoch_loss:.4f}")

    # Save the trained model and scaler
    model.save("model.keras")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved.")

    # Plot the losses
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, marker="o")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss per Batch")
    plt.grid()
    plt.show()

# Preprocess Data
def preprocess_data(df, train=True):
    # Separate the features and target variable
    X = df.drop("actualPP", axis=1)
    y = df["actualPP"]

    # Scale the features
    if train:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y

# Train the model using all data
train_model()
