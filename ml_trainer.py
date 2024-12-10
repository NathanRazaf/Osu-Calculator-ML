import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from data_processor import get_data

Sequential = tf.keras.Sequential
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.Adam
Input = tf.keras.Input

def create_model(architecture):
    """
    Creates a model based on an architecture configuration dictionary
    
    architecture = {
        'layers': [units, ...],
        'learning_rate': float,
        'activation': str
    }
    """
    inputs = Input(shape=(20,))
    x = inputs
    
    for units in architecture['layers']:
        x = Dense(units, activation=architecture['activation'])(x)
    
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=architecture['learning_rate']), 
        loss="mse", 
        metrics=["mae"]
    )
    return model

def train_architectures(architectures):
    # Load all training data
    all_data = pd.concat(get_data(), ignore_index=True)
    X = all_data.drop("actualPP", axis=1).values
    y = all_data["actualPP"].values
    
    # Scale all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Results storage
    architecture_results = {}
    
    # Test each architecture
    for arch_name, arch_config in architectures.items():
        print(f"\n=== Training Architecture: {arch_name} ===")
        
        # Create and train model
        model = create_model(arch_config)
        history = model.fit(
            X_scaled, 
            y,
            epochs=arch_config.get('epochs', 15),
            batch_size=arch_config.get('batch_size', 16),
            verbose=1,
            shuffle=True
        )
        
        # Save model and scaler
        model.save(f"{arch_name}_model.keras")
        joblib.dump(scaler, f"{arch_name}_scaler.pkl")
        
        # Store training history
        architecture_results[arch_name] = {
            'final_loss': history.history["loss"][-1],
            'final_mae': history.history["mae"][-1],
            'training_history': history.history
        }
        
        print(f"\n{arch_name} Final Training Metrics:")
        print(f"Loss: {history.history['loss'][-1]:.4f}")
        print(f"MAE: {history.history['mae'][-1]:.4f}")
    
    return architecture_results

# Example architectures to test
architectures = {
    # 'baseline': {
    #     'layers': [84, 42],
    #     'learning_rate': 0.0007,
    #     'activation': 'relu',
    #     'epochs': 15,
    #     'batch_size': 16
    # },
    # 'deeper': {
    #     'layers': [128, 64, 32],
    #     'learning_rate': 0.0007,
    #     'activation': 'relu',
    #     'epochs': 20,
    #     'batch_size': 32
    # },
    # 'wider': {
    #     'layers': [256, 128],
    #     'learning_rate': 0.00035,
    #     'activation': 'elu',
    #     'epochs': 25,
    #     'batch_size': 64
    # },
    # 'very_deep': {
    #     'layers': [256, 128, 64, 32],
    #     'learning_rate': 0.0006,
    #     'activation': 'relu',
    #     'epochs': 25,
    #     'batch_size': 32
    # },
    'ultra_deep': {
        'layers': [1024, 512, 256, 128, 64, 32],
        'learning_rate': 0.00005,
        'activation': 'relu',
        'epochs': 64,
        'batch_size': 32
    }
}

# Run the training
results = train_architectures(architectures)