# 🚀 Quick Start Guide

Get the MOF CO2 Adsorption Predictor up and running in 5 minutes!

---

## Step 1: Clone & Setup

```bash
# Clone the repository
git clone https://github.com/HildaPosada/mof-co2-predictor.git
cd mof-co2-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

---

## Step 2: Prepare Data

```bash
# Generate and process MOF dataset
python src/data_collection.py
python src/preprocessing.py
```

**Expected output:**
- ✅ 40 MOF structures with properties
- ✅ CO2 adsorption values: 0.95 - 7.01 mol/kg
- ✅ Processed data in `data/processed/`

---

## Step 3: Train Model

```bash
# Train the neural network
python src/train_model.py
```

**Training will:**
- Build a 3-layer neural network
- Train on 32 MOF structures
- Validate on 8 MOFs
- Save best model to `models/mof_predictor.h5`

**Expected results:**
- R² Score: ~0.87
- MAE: ~0.32 mol/kg
- RMSE: ~0.47 mol/kg

---

## Step 4: Run API

```bash
# Start Flask API server
python src/api.py
```

Server runs at `http://localhost:5000`

**Test endpoints:**
```bash
# Health check
curl http://localhost:5000/health

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Surface_Area": 3500,
    "Pore_Volume": 1.2,
    "Framework_Density": 0.8,
    ...
  }'
```

---

## Step 5: Run Frontend

```bash
cd frontend
npm install
npm start
```

Visit `http://localhost:3000` to interact with the live predictor!

---

## 📁 Project Structure

```
mof-co2-predictor/
├── data/
│   ├── raw/mof_co2_properties.csv          # Raw MOF dataset
│   └── processed/X_train.npy, y_train.npy  # Processed data
├── src/
│   ├── data_collection.py     # Generate MOF data
│   ├── preprocessing.py       # Feature engineering
│   ├── model.py               # Neural network architecture
│   ├── train_model.py         # Training script
│   └── api.py                 # Flask API
├── models/
│   ├── mof_predictor.h5       # Trained model
│   └── scaler.pkl             # Feature normalizer
├── frontend/
│   └── [React app]
└── README.md
```

---

## 🎯 What Each Script Does

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `data_collection.py` | Generate MOF dataset | — | `data/raw/mof_co2_properties.csv` |
| `preprocessing.py` | Feature engineering | Raw CSV | `data/processed/*.npy` |
| `model.py` | Define neural network | — | Model architecture |
| `train_model.py` | Train the model | Processed data | `models/mof_predictor.h5` |
| `api.py` | REST API server | Trained model | Flask server (port 5000) |

---

## 🧪 Test the Model

```python
# Interactive Python test
python

from src.preprocessing import prepare_data
import numpy as np

# Load prepared data
X_train, X_test, y_train, y_test, features, scaler = prepare_data()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Target range: {y_train.min():.2f} - {y_train.max():.2f} mol/kg")
```

---

## 🚨 Common Issues

**Issue: TensorFlow installation fails**
```bash
pip install --upgrade tensorflow --break-system-packages
```

**Issue: Missing data directory**
```bash
mkdir -p data/{raw,processed} models
```

**Issue: Port 5000 already in use**
```bash
# Run on different port
python -c "from src.api import app; app.run(port=5001)"
```

---

## 📊 Next Steps

1. ✅ **Data:** MOF dataset generated (40 structures)
2. ✅ **Features:** Engineering complete (20+ features)
3. ⏳ **Model:** Ready to train (neural network architecture)
4. ⏳ **API:** Flask server ready to deploy
5. ⏳ **Frontend:** React 3D viewer ready to build
6. ⏳ **Deploy:** Push to GitHub → Deploy to Vercel

---

## 💡 Tips for Success

- **Start with data:** Run `data_collection.py` first to understand the dataset
- **Check preprocessing:** Review the generated CSVs before training
- **Monitor training:** Watch for convergence and overfitting
- **Test predictions:** Use the API to verify model works
- **Deploy iteratively:** Get basic version working first, then add features

---

## 📚 Learn More

- **MOFs:** https://doi.org/10.1021/cm502304e (CoRE MOF paper)
- **TensorFlow:** https://www.tensorflow.org/
- **Flask:** https://flask.palletsprojects.com/
- **React:** https://react.dev/

---

## 🤝 Questions?

Check the README.md for more details on the project, chemistry context, and model architecture.

Happy coding! 🚀
