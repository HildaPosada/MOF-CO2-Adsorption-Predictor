"""
Flask REST API for MOF CO2 Adsorption Prediction

This API provides endpoints for making predictions using the trained neural network model.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables for model and scaler
model = None
scaler = None
label_encoder = None
feature_names = None

def load_model_and_scaler():
    """Load the trained model and preprocessing objects"""
    global model, scaler, label_encoder, feature_names
    
    try:
        # Load model
        model_path = 'models/mof_predictor.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Load scaler
        scaler_path = 'models/scaler.pkl'
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from {scaler_path}")
        
        # Load label encoder
        encoder_path = 'models/metal_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            print(f"✅ Label encoder loaded from {encoder_path}")
        
        # Load feature names
        feature_path = 'data/processed/feature_names.txt'
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            print(f"✅ Feature names loaded: {len(feature_names)} features")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expected JSON input:
    {
        "Surface_Area": 3500,
        "Pore_Volume": 1.2,
        "Framework_Density": 0.8,
        "Pore_Diameter": 12.5,
        "Void_Fraction": 0.65,
        "Largest_Cavity_Diameter": 18.0,
        "Pore_Limiting_Diameter": 8.0,
        "Metal_Content": 25.0,
        "Organic_Content": 75.0,
        "Has_Amino_Group": 1,
        "Has_Carboxylate": 1,
        "Has_Hydroxyl": 0,
        "Metal_Type": "Zn"
    }
    
    Returns:
    {
        "prediction": 4.52,
        "unit": "mol/kg"
    }
    """
    
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract base features
        try:
            surface_area = float(data['Surface_Area'])
            pore_volume = float(data['Pore_Volume'])
            framework_density = float(data['Framework_Density'])
            pore_diameter = float(data['Pore_Diameter'])
            void_fraction = float(data['Void_Fraction'])
            largest_cavity = float(data['Largest_Cavity_Diameter'])
            pore_limiting = float(data['Pore_Limiting_Diameter'])
            metal_content = float(data['Metal_Content'])
            organic_content = float(data['Organic_Content'])
            has_amino = int(data['Has_Amino_Group'])
            has_carboxylate = int(data['Has_Carboxylate'])
            has_hydroxyl = int(data['Has_Hydroxyl'])
            metal_type = str(data['Metal_Type'])
        except KeyError as e:
            return jsonify({'error': f'Missing required field: {str(e)}'}), 400
        except ValueError as e:
            return jsonify({'error': f'Invalid value: {str(e)}'}), 400
        
        # Encode metal type
        if label_encoder:
            try:
                metal_encoded = label_encoder.transform([metal_type])[0]
            except:
                # Use default if metal type not recognized
                metal_encoded = 0
        else:
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
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        
        # Return result
        return jsonify({
            'prediction': float(prediction),
            'unit': 'mol/kg',
            'input_features': {
                'Surface_Area': surface_area,
                'Pore_Volume': pore_volume,
                'Framework_Density': framework_density,
                'Metal_Type': metal_type
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple MOF structures.
    
    Expected JSON input:
    {
        "mofs": [
            { ... MOF 1 properties ... },
            { ... MOF 2 properties ... },
            ...
        ]
    }
    """
    
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'mofs' not in data:
            return jsonify({'error': 'No MOF data provided'}), 400
        
        predictions = []
        
        for mof_data in data['mofs']:
            # Use the single prediction logic for each MOF
            # (In production, this should be optimized for batch processing)
            request.json = mof_data
            result = predict()
            predictions.append(result[0].get_json())
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature_info', methods=['GET'])
def feature_info():
    """Return information about required features"""
    
    feature_descriptions = {
        'Surface_Area': 'BET surface area (m²/g)',
        'Pore_Volume': 'Pore volume (cm³/g)',
        'Framework_Density': 'Framework density (g/cm³)',
        'Pore_Diameter': 'Average pore diameter (Å)',
        'Void_Fraction': 'Void fraction (0-1)',
        'Largest_Cavity_Diameter': 'Largest cavity diameter (Å)',
        'Pore_Limiting_Diameter': 'Pore limiting diameter (Å)',
        'Metal_Content': 'Metal content (% by weight)',
        'Organic_Content': 'Organic linker content (% by weight)',
        'Has_Amino_Group': 'Presence of amino groups (0 or 1)',
        'Has_Carboxylate': 'Presence of carboxylate groups (0 or 1)',
        'Has_Hydroxyl': 'Presence of hydroxyl groups (0 or 1)',
        'Metal_Type': 'Metal type (Zn, Cu, Zr, Al, Cr)'
    }
    
    return jsonify({
        'required_features': feature_descriptions,
        'total_features': len(feature_descriptions),
        'derived_features': [
            'Surface_to_Volume_Ratio',
            'Density_Volume_Product',
            'Cavity_to_Pore_Ratio',
            'Accessibility'
        ]
    }), 200

def main():
    """Start the Flask API server"""
    
    print("\n" + "=" * 60)
    print("🚀 MOF CO2 Adsorption Predictor API")
    print("=" * 60)
    
    # Load model and scaler
    if not load_model_and_scaler():
        print("\n❌ Failed to load model!")
        print("   Please ensure you have trained the model first:")
        print("   python src/train_model.py")
        return
    
    print("\n" + "=" * 60)
    print("✅ API Server Ready!")
    print("=" * 60)
    print("\n📍 Available Endpoints:")
    print("   - GET  /health          - Health check")
    print("   - POST /predict         - Single prediction")
    print("   - POST /batch_predict   - Batch predictions")
    print("   - GET  /feature_info    - Feature information")
    
    print("\n🌐 Server running at: http://localhost:5000")
    print("\n💡 Test with:")
    print("   curl http://localhost:5000/health")
    
    print("\n" + "=" * 60)
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()
