"""
Neural Network Model Architecture

Defines the deep learning model for predicting CO2 adsorption capacity in MOFs.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def build_model(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3, l2_reg=0.001):
    """
    Build a feedforward neural network for regression.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_layers : list
        List of hidden layer sizes (default: [128, 64, 32])
    dropout_rate : float
        Dropout rate for regularization (default: 0.3)
    l2_reg : float
        L2 regularization coefficient (default: 0.001)
    
    Returns:
    --------
    keras.Model
        Compiled neural network model
    """
    
    model = keras.Sequential(name='MOF_CO2_Predictor')
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,), name='input'))
    
    # Hidden layers with batch normalization, dropout, and L2 regularization
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer (regression)
    model.add(layers.Dense(1, activation='linear', name='output'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=[
            'mae',  # Mean Absolute Error
            keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    
    return model

def get_callbacks(model_path='models/mof_predictor.h5', patience=20):
    """
    Define training callbacks for model optimization.
    
    Parameters:
    -----------
    model_path : str
        Path to save the best model
    patience : int
        Number of epochs to wait before early stopping
    
    Returns:
    --------
    list
        List of Keras callbacks
    """
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def print_model_summary(model):
    """Print detailed model architecture"""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    model.summary()
    print("=" * 60)
    
    # Count parameters
    total_params = model.count_params()
    print(f"\n📊 Total Parameters: {total_params:,}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Example usage
    print("🧠 Neural Network Architecture for MOF CO2 Prediction")
    print("=" * 60)
    
    # Build example model
    example_model = build_model(input_dim=17)
    print_model_summary(example_model)
    
    print("✅ Model architecture defined successfully!")
    print("\nTo train the model, run:")
    print("   python src/train_model.py")
