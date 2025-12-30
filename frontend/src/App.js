import React, { useState } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    Surface_Area: 3500,
    Pore_Volume: 1.2,
    Framework_Density: 0.8,
    Pore_Diameter: 12.5,
    Void_Fraction: 0.65,
    Largest_Cavity_Diameter: 18.0,
    Pore_Limiting_Diameter: 8.0,
    Metal_Content: 25.0,
    Organic_Content: 75.0,
    Has_Amino_Group: 1,
    Has_Carboxylate: 1,
    Has_Hydroxyl: 0,
    Metal_Type: 'Zn'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.startsWith('Has_') ? parseInt(value) : value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setPrediction(response.data.prediction);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to make prediction. Ensure the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <h1>🔬 MOF CO₂ Predictor</h1>
        <p className="subtitle">
          Predict CO₂ adsorption capacity using machine learning
        </p>

        <div className="form-container">
          <div className="input-grid">
            <div className="input-group">
              <label>Surface Area (m²/g)</label>
              <input
                type="number"
                name="Surface_Area"
                value={formData.Surface_Area}
                onChange={handleInputChange}
                step="100"
              />
            </div>

            <div className="input-group">
              <label>Pore Volume (cm³/g)</label>
              <input
                type="number"
                name="Pore_Volume"
                value={formData.Pore_Volume}
                onChange={handleInputChange}
                step="0.1"
              />
            </div>

            <div className="input-group">
              <label>Framework Density (g/cm³)</label>
              <input
                type="number"
                name="Framework_Density"
                value={formData.Framework_Density}
                onChange={handleInputChange}
                step="0.1"
              />
            </div>

            <div className="input-group">
              <label>Pore Diameter (Å)</label>
              <input
                type="number"
                name="Pore_Diameter"
                value={formData.Pore_Diameter}
                onChange={handleInputChange}
                step="0.5"
              />
            </div>

            <div className="input-group">
              <label>Void Fraction (0-1)</label>
              <input
                type="number"
                name="Void_Fraction"
                value={formData.Void_Fraction}
                onChange={handleInputChange}
                step="0.05"
                min="0"
                max="1"
              />
            </div>

            <div className="input-group">
              <label>Largest Cavity Diameter (Å)</label>
              <input
                type="number"
                name="Largest_Cavity_Diameter"
                value={formData.Largest_Cavity_Diameter}
                onChange={handleInputChange}
                step="1"
              />
            </div>

            <div className="input-group">
              <label>Pore Limiting Diameter (Å)</label>
              <input
                type="number"
                name="Pore_Limiting_Diameter"
                value={formData.Pore_Limiting_Diameter}
                onChange={handleInputChange}
                step="0.5"
              />
            </div>

            <div className="input-group">
              <label>Metal Content (%)</label>
              <input
                type="number"
                name="Metal_Content"
                value={formData.Metal_Content}
                onChange={handleInputChange}
                step="1"
              />
            </div>

            <div className="input-group">
              <label>Organic Content (%)</label>
              <input
                type="number"
                name="Organic_Content"
                value={formData.Organic_Content}
                onChange={handleInputChange}
                step="1"
              />
            </div>

            <div className="input-group">
              <label>Has Amino Group</label>
              <select
                name="Has_Amino_Group"
                value={formData.Has_Amino_Group}
                onChange={handleInputChange}
              >
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="input-group">
              <label>Has Carboxylate</label>
              <select
                name="Has_Carboxylate"
                value={formData.Has_Carboxylate}
                onChange={handleInputChange}
              >
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="input-group">
              <label>Has Hydroxyl</label>
              <select
                name="Has_Hydroxyl"
                value={formData.Has_Hydroxyl}
                onChange={handleInputChange}
              >
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="input-group">
              <label>Metal Type</label>
              <select
                name="Metal_Type"
                value={formData.Metal_Type}
                onChange={handleInputChange}
              >
                <option value="Zn">Zinc (Zn)</option>
                <option value="Cu">Copper (Cu)</option>
                <option value="Zr">Zirconium (Zr)</option>
                <option value="Al">Aluminum (Al)</option>
                <option value="Cr">Chromium (Cr)</option>
              </select>
            </div>
          </div>

          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? '🔄 Predicting...' : '🚀 Predict CO₂ Adsorption'}
          </button>
        </div>

        {prediction !== null && (
          <div className="result-container">
            <h2>📊 Prediction Result</h2>
            <div className="result-value">
              {prediction.toFixed(2)} mol/kg
            </div>
            <p>Predicted CO₂ adsorption capacity at 298K, 1 bar</p>
          </div>
        )}

        {error && (
          <div className="error">
            <strong>❌ Error:</strong> {error}
          </div>
        )}

        <div style={{ marginTop: '40px', opacity: 0.8, fontSize: '0.9rem' }}>
          <p>
            💡 <strong>Tip:</strong> Make sure the Flask API is running on port 5000
          </p>
          <p>
            Run: <code style={{ background: 'rgba(0,0,0,0.2)', padding: '4px 8px', borderRadius: '4px' }}>
              python src/api.py
            </code>
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
