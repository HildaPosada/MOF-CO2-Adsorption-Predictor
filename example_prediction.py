#!/usr/bin/env python
"""
Example script demonstrating how to use the trained model for predictions.

This script shows how to:
1. Load a trained model
2. Prepare input features
3. Make predictions
4. Interpret results
"""

import numpy as np
import joblib
import tensorflow as tf
import os

def load_model_components():
    """Load the trained model, scaler, and encoders"""
    
    print("Loading model components...")
    
    # Check if model exists
    if not os.path.exists('models/mof_predictor.h5'):
        raise FileNotFoundError(
            "Trained model not found. Please run 'python src/train_model.py' first."
        )
    
    # Load model
    model = tf.keras.models.load_model('models/mof_predictor.h5')
    print("✅ Model loaded")
    
    # Load scaler
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler loaded")
    
    # Load label encoder
    label_encoder = joblib.load('models/metal_encoder.pkl')
    print("✅ Label encoder loaded")
    
    return model, scaler, label_encoder

def prepare_features(mof_properties, label_encoder):
    """
    Prepare feature vector from MOF properties.
    
    Parameters:
    -----------
    mof_properties : dict
        Dictionary containing MOF properties
    label_encoder : LabelEncoder
        Encoder for metal types
    
    Returns:
    --------
    np.ndarray
        Feature vector ready for prediction
    """
    
    # Extract base features
    surface_area = mof_properties['Surface_Area']
    pore_volume = mof_properties['Pore_Volume']
    framework_density = mof_properties['Framework_Density']
    pore_diameter = mof_properties['Pore_Diameter']
    void_fraction = mof_properties['Void_Fraction']
    largest_cavity = mof_properties['Largest_Cavity_Diameter']
    pore_limiting = mof_properties['Pore_Limiting_Diameter']
    metal_content = mof_properties['Metal_Content']
    organic_content = mof_properties['Organic_Content']
    has_amino = mof_properties['Has_Amino_Group']
    has_carboxylate = mof_properties['Has_Carboxylate']
    has_hydroxyl = mof_properties['Has_Hydroxyl']
    
    # Encode metal type
    metal_type = mof_properties['Metal_Type']
    try:
        metal_encoded = label_encoder.transform([metal_type])[0]
    except:
        print(f"Warning: Unknown metal type '{metal_type}', using default")
        metal_encoded = 0
    
    # Calculate derived features
    surface_to_volume = surface_area / (pore_volume + 1e-6)
    density_volume = framework_density * pore_volume
    cavity_to_pore = largest_cavity / (pore_limiting + 1e-6)
    accessibility = void_fraction * pore_diameter
    
    # Construct feature vector (must match training order)
    features = np.array([[
        surface_area, pore_volume, framework_density,
        pore_diameter, void_fraction, largest_cavity,
        pore_limiting, metal_content, organic_content,
        has_amino, has_carboxylate, has_hydroxyl,
        metal_encoded, surface_to_volume, density_volume,
        cavity_to_pore, accessibility
    ]])
    
    return features

def predict_co2_adsorption(mof_properties):
    """
    Predict CO2 adsorption capacity for a MOF structure.
    
    Parameters:
    -----------
    mof_properties : dict
        Dictionary containing MOF properties
    
    Returns:
    --------
    float
        Predicted CO2 adsorption capacity (mol/kg)
    """
    
    # Load model components
    model, scaler, label_encoder = load_model_components()
    
    # Prepare features
    features = prepare_features(mof_properties, label_encoder)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled, verbose=0)[0][0]
    
    return prediction

def main():
    """Example usage"""
    
    print("\n" + "=" * 60)
    print("🔬 MOF CO2 Adsorption Prediction Example")
    print("=" * 60)
    
    # Example MOF structures
    examples = [
        {
            'name': 'High-Performance MOF',
            'properties': {
                'Surface_Area': 5000,
                'Pore_Volume': 2.0,
                'Framework_Density': 0.5,
                'Pore_Diameter': 15.0,
                'Void_Fraction': 0.8,
                'Largest_Cavity_Diameter': 25.0,
                'Pore_Limiting_Diameter': 10.0,
                'Metal_Content': 20.0,
                'Organic_Content': 80.0,
                'Has_Amino_Group': 1,
                'Has_Carboxylate': 1,
                'Has_Hydroxyl': 0,
                'Metal_Type': 'Zn'
            }
        },
        {
            'name': 'Dense MOF',
            'properties': {
                'Surface_Area': 1500,
                'Pore_Volume': 0.5,
                'Framework_Density': 1.5,
                'Pore_Diameter': 6.0,
                'Void_Fraction': 0.4,
                'Largest_Cavity_Diameter': 8.0,
                'Pore_Limiting_Diameter': 4.0,
                'Metal_Content': 35.0,
                'Organic_Content': 65.0,
                'Has_Amino_Group': 0,
                'Has_Carboxylate': 1,
                'Has_Hydroxyl': 0,
                'Metal_Type': 'Cu'
            }
        },
        {
            'name': 'Moderate MOF',
            'properties': {
                'Surface_Area': 3500,
                'Pore_Volume': 1.2,
                'Framework_Density': 0.8,
                'Pore_Diameter': 12.5,
                'Void_Fraction': 0.65,
                'Largest_Cavity_Diameter': 18.0,
                'Pore_Limiting_Diameter': 8.0,
                'Metal_Content': 25.0,
                'Organic_Content': 75.0,
                'Has_Amino_Group': 1,
                'Has_Carboxylate': 1,
                'Has_Hydroxyl': 0,
                'Metal_Type': 'Zn'
            }
        }
    ]
    
    # Make predictions for each example
    for example in examples:
        print(f"\n{'─' * 60}")
        print(f"📊 {example['name']}")
        print(f"{'─' * 60}")
        
        # Display key properties
        props = example['properties']
        print(f"Surface Area: {props['Surface_Area']} m²/g")
        print(f"Pore Volume: {props['Pore_Volume']} cm³/g")
        print(f"Framework Density: {props['Framework_Density']} g/cm³")
        print(f"Metal Type: {props['Metal_Type']}")
        
        # Predict
        try:
            prediction = predict_co2_adsorption(props)
            print(f"\n🎯 Predicted CO₂ Adsorption: {prediction:.2f} mol/kg")
            
            # Interpretation
            if prediction > 5.0:
                rating = "Excellent"
            elif prediction > 3.5:
                rating = "Good"
            elif prediction > 2.0:
                rating = "Moderate"
            else:
                rating = "Low"
            
            print(f"   Rating: {rating}")
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✨ Predictions complete!")
    print("\nTo make your own predictions:")
    print("1. Start the API: python src/api.py")
    print("2. Use the web interface: cd frontend && npm start")
    print("3. Or send POST requests to http://localhost:5000/predict")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
