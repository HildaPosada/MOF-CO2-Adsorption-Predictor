"""
Model Training Script

Train the neural network on preprocessed MOF data and evaluate performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model import build_model, get_callbacks, print_model_summary

def load_preprocessed_data():
    """Load preprocessed training and test data"""
    
    print("📂 Loading preprocessed data...")
    
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"   ✅ Training set: {X_train.shape}")
    print(f"   ✅ Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, epochs=200, batch_size=8):
    """
    Train the neural network model.
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test targets
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
    
    Returns:
    --------
    model, history
        Trained model and training history
    """
    
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK")
    print("=" * 60)
    
    # Build model
    input_dim = X_train.shape[1]
    model = build_model(input_dim=input_dim)
    print_model_summary(model)
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    print("🚀 Starting training...")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Validation split: 20%")
    print("\n" + "-" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    """
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n📊 Test Set Performance:")
    print(f"   - R² Score:  {r2:.4f}")
    print(f"   - MAE:       {mae:.4f} mol/kg")
    print(f"   - RMSE:      {rmse:.4f} mol/kg")
    print(f"   - MAPE:      {mape:.2f}%")
    
    print(f"\n🎯 Prediction Examples:")
    print(f"   {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print(f"   {'-'*36}")
    for i in range(min(5, len(y_test))):
        error = abs(y_test[i] - y_pred[i])
        print(f"   {y_test[i]:<12.3f} {y_pred[i]:<12.3f} {error:<12.3f}")
    
    print("=" * 60)
    
    return y_pred, r2, mae, rmse

def plot_training_history(history):
    """
    Plot training and validation loss curves.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE (mol/kg)', fontsize=12)
    axes[1].set_title('Mean Absolute Error During Training', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Training history plot saved to: models/training_history.png")

def plot_predictions(y_test, y_pred):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_test : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    """
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual CO₂ Adsorption (mol/kg)', fontsize=12)
    plt.ylabel('Predicted CO₂ Adsorption (mol/kg)', fontsize=12)
    plt.title('Model Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('models/predictions.png', dpi=300, bbox_inches='tight')
    print(f"📊 Prediction plot saved to: models/predictions.png")

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 60)
    print("🧠 MOF CO2 ADSORPTION PREDICTOR - MODEL TRAINING")
    print("=" * 60)
    
    # Check if preprocessed data exists
    if not os.path.exists('data/processed/X_train.npy'):
        print("\n❌ Preprocessed data not found!")
        print("   Please run: python src/preprocessing.py")
        return
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train model
    model, history = train_model(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    y_pred, r2, mae, rmse = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_training_history(history)
    plot_predictions(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("✨ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Saved Files:")
    print(f"   - models/mof_predictor.h5 (trained model)")
    print(f"   - models/training_history.png")
    print(f"   - models/predictions.png")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review the plots in models/ directory")
    print(f"   2. Start the API: python src/api.py")
    print(f"   3. Test predictions via REST API")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
