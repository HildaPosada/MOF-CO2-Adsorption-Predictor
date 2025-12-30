"""
MOF Data Preprocessing Pipeline

This script handles feature engineering, data cleaning, and preparation
for the neural network model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_raw_data():
    """Load the raw MOF dataset"""
    data_path = 'data/raw/mof_co2_properties.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Raw data not found at {data_path}. "
            "Please run 'python src/data_collection.py' first."
        )
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} MOF structures from {data_path}")
    return df

def engineer_features(df):
    """
    Create derived features based on domain knowledge.
    
    Additional features:
    - Surface_to_Volume_Ratio: Indicator of accessibility
    - Density_Volume_Product: Structural compactness
    - Cavity_to_Pore_Ratio: Pore network complexity
    """
    
    print("🔧 Engineering features...")
    
    df_processed = df.copy()
    
    # Derived features
    df_processed['Surface_to_Volume_Ratio'] = (
        df_processed['Surface_Area'] / (df_processed['Pore_Volume'] + 1e-6)
    )
    
    df_processed['Density_Volume_Product'] = (
        df_processed['Framework_Density'] * df_processed['Pore_Volume']
    )
    
    df_processed['Cavity_to_Pore_Ratio'] = (
        df_processed['Largest_Cavity_Diameter'] / 
        (df_processed['Pore_Limiting_Diameter'] + 1e-6)
    )
    
    # Accessibility metric
    df_processed['Accessibility'] = (
        df_processed['Void_Fraction'] * df_processed['Pore_Diameter']
    )
    
    # Encode categorical variable (Metal_Type)
    label_encoder = LabelEncoder()
    df_processed['Metal_Type_Encoded'] = label_encoder.fit_transform(df_processed['Metal_Type'])
    
    # Save the encoder for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoder, 'models/metal_encoder.pkl')
    print(f"   - Metal type encoder saved")
    
    print(f"   - Created {4} derived features")
    print(f"   - Encoded categorical variable: Metal_Type")
    
    return df_processed, label_encoder

def prepare_data(test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names, scaler
    """
    
    print("=" * 60)
    print("MOF Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_raw_data()
    
    # Engineer features
    df_processed, label_encoder = engineer_features(df)
    
    # Select feature columns (exclude name and target)
    feature_columns = [
        'Surface_Area', 'Pore_Volume', 'Framework_Density', 
        'Pore_Diameter', 'Void_Fraction', 'Largest_Cavity_Diameter',
        'Pore_Limiting_Diameter', 'Metal_Content', 'Organic_Content',
        'Has_Amino_Group', 'Has_Carboxylate', 'Has_Hydroxyl',
        'Metal_Type_Encoded', 'Surface_to_Volume_Ratio',
        'Density_Volume_Product', 'Cavity_to_Pore_Ratio', 'Accessibility'
    ]
    
    # Prepare features and target
    X = df_processed[feature_columns].values
    y = df_processed['CO2_Adsorption_mol_kg'].values
    
    print(f"\n📊 Feature Matrix Shape: {X.shape}")
    print(f"📊 Target Vector Shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n✂️  Train-Test Split:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Standardize features (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n🔄 Feature Scaling:")
    print(f"   - Method: StandardScaler (zero mean, unit variance)")
    print(f"   - Features normalized: {len(feature_columns)}")
    
    # Save preprocessed data
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_columns))
    
    print(f"\n💾 Saved Preprocessed Data:")
    print(f"   - data/processed/X_train.npy")
    print(f"   - data/processed/X_test.npy")
    print(f"   - data/processed/y_train.npy")
    print(f"   - data/processed/y_test.npy")
    print(f"   - models/scaler.pkl")
    print(f"   - data/processed/feature_names.txt")
    
    print(f"\n📈 Target Statistics:")
    print(f"   - Training mean: {y_train.mean():.2f} mol/kg")
    print(f"   - Training std: {y_train.std():.2f} mol/kg")
    print(f"   - Training range: {y_train.min():.2f} - {y_train.max():.2f} mol/kg")
    
    print("\n✨ Preprocessing complete! Ready for model training.")
    print("   Run: python src/train_model.py")
    print("=" * 60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, scaler

def main():
    """Main execution function"""
    prepare_data()

if __name__ == "__main__":
    main()
