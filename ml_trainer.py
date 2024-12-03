import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from data_processor import get_data
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Initialize the StandardScaler
scaler = StandardScaler()

# Initialize the XGBRegressor
model = XGBRegressor(
    n_estimators=100,       # Number of trees
    learning_rate=0.1,      # Shrinkage rate
    max_depth=6,            # Maximum tree depth
    random_state=42         # Reproducibility
)

# Track MSE per batch
batch_mse = []

def train_model():
    test_data = []  # Collect test samples
    for batch_index, batch_df in enumerate(get_data()):
        # Split the batch into train and test
        train_df, test_df = train_test_split(batch_df, test_size=0.2, random_state=42)

        # Train on the train part of the batch
        X_train, y_train = preprocess_data(train_df, train=True)
        model.fit(X_train, y_train)

        # Evaluate on the test part of the batch
        X_test, y_test = preprocess_data(test_df, train=False)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Store MSE for this batch
        batch_mse.append(mse)

        # Collect test data for final evaluation
        test_data.append(test_df)

        # Log progress
        print(f"Batch {batch_index + 1} - MSE: {mse}")

    # Combine all test batches into one DataFrame
    test_df = pd.concat(test_data)
    return test_df  # Return test data for final evaluation


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


# Train the model and record MSE for each batch
test_df = train_model()

# Plot the MSE over batches
plt.plot(range(1, len(batch_mse) + 1), batch_mse, marker="o")
plt.xlabel("Batch Number")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE over Training Batches")
plt.grid()
plt.show()
