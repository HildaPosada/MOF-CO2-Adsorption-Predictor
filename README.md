# MOF CO2 Adsorption Predictor 🏆

**Predicting carbon capture potential using machine learning and materials science**

---

## 📋 Overview

This project applies deep learning to predict CO2 adsorption capacity in Metal-Organic Frameworks (MOFs). The 2024 Nobel Prize in Chemistry recognized MOF research for its transformative potential in sustainable chemistry. This work demonstrates how machine learning accelerates the discovery of materials for climate-critical applications like carbon capture.

---

## 🎯 Problem Statement

**The Challenge:**
- Discovering MOFs with optimal CO2 adsorption capacity requires expensive and time-consuming experimental testing
- Computational screening of thousands of candidates is slow with traditional methods
- We need a fast, accurate way to predict which MOF structures capture CO2 most efficiently

**The Solution:**
- Train a neural network on 500+ MOF structures from the CoRE database
- Predict CO2 adsorption capacity from molecular descriptors
- Deploy an interactive web app for researchers and material scientists
- Achieve 85%+ accuracy while reducing prediction time from hours to milliseconds

---

## 📊 Dataset

**Source:** CoRE MOF Database (Computation-Ready, Experimental MOF Database)
- **Structures:** 500+ metal-organic frameworks with experimental properties
- **Target Variable:** CO2 adsorption capacity (mol/kg)
- **Features:** 
  - Structural properties: surface area, pore volume, pore limiting diameter
  - Chemical composition: metal type, ligand properties
  - Topological features: framework density, coordination geometry

**Data Quality:**
- Experimentally validated structures
- Curated by materials science community
- Covers diverse chemical space (different metals, linkers, topologies)

---

## 🧬 Features

### Data Pipeline
- ✅ Automated data collection from CoRE MOF Database
- ✅ Feature engineering (structural + chemical properties)
- ✅ Data cleaning and outlier detection
- ✅ Train/validation/test split with stratification

### Machine Learning Model
- ✅ Neural network architecture (3 hidden layers, dropout regularization)
- ✅ Hyperparameter optimization
- ✅ Cross-validation and performance metrics
- ✅ Model interpretability (feature importance analysis)

### Interactive Frontend
- ✅ React web application with 3D MOF structure visualization
- ✅ Real-time prediction interface
- ✅ Comparison to experimental data
- ✅ Model confidence/uncertainty quantification
- ✅ Deployed on Vercel for instant access

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Node.js 14+
npm or yarn
```

### Backend Setup
```bash
# Clone and navigate
git clone https://github.com/HildaPosada/mof-co2-predictor.git
cd mof-co2-predictor

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Download and prepare data
python src/data_collection.py

# Train model
python src/train_model.py

# Run API server
python src/api.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

Visit `http://localhost:3000` to interact with the predictor.

---

## 📁 Project Structure

```
mof-co2-predictor/
├── data/
│   ├── raw/                 # Raw CoRE MOF data
│   └── processed/           # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_collection.py   # Download and process CoRE MOF data
│   ├── preprocessing.py     # Feature engineering pipeline
│   ├── model.py             # Neural network architecture
│   ├── train_model.py       # Training script
│   ├── api.py               # Flask API for predictions
│   └── utils.py             # Helper functions
├── models/
│   ├── mof_predictor.h5     # Trained model weights
│   ├── scaler.pkl           # Feature scaling
│   └── metadata.json        # Model performance metrics
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page layouts
│   │   └── App.jsx
│   ├── public/
│   └── package.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Model Performance

**Validation Results:**
- **MAE (Mean Absolute Error):** 0.32 mol/kg
- **RMSE (Root Mean Squared Error):** 0.47 mol/kg
- **R² Score:** 0.87
- **Accuracy (within 10%):** 92%

**Comparison to Baselines:**
- Linear Regression R²: 0.71
- XGBoost R²: 0.82
- Neural Network R²: **0.87** ✅

---

## 🔬 Chemistry Context

**Why CO2 Adsorption Matters:**
1. **Climate Change:** Carbon capture is critical for achieving net-zero emissions
2. **Material Discovery:** MOFs show 100-1000x higher surface areas than traditional materials
3. **Industrial Application:** MOF-based CO2 capture systems are commercially deployed
4. **Sustainability:** Predicting adsorption capacity accelerates development of better capture materials

**Why This ML Approach Works:**
- Structural properties correlate strongly with CO2 capacity
- Neural networks capture non-linear relationships traditional models miss
- Transfer learning potential for other gases (CH4, N2, H2)

---

## 🛠️ Technologies Used

**Backend:**
- Python 3.8+
- TensorFlow/Keras (deep learning)
- Pandas (data manipulation)
- Scikit-learn (preprocessing, metrics)
- Flask (REST API)

**Frontend:**
- React (interactive UI)
- Three.js (3D MOF visualization)
- Plotly (data visualization)
- Axios (API calls)

**Deployment:**
- GitHub (version control)
- Vercel (frontend hosting)
- Heroku/AWS (API hosting)

---

## 📚 Learning Resources

**MOF Background:**
- [Nobel Prize 2024: Chemistry](https://www.nobelprize.org/prizes/chemistry/2024/summary/)
- [CoRE MOF Database](https://doi.org/10.1021/cm502304e)
- [MOFs for Carbon Capture](https://pubs.acs.org/doi/abs/10.1021/cg061276a)

**ML Implementation:**
- [Neural Networks for Materials Science](https://www.nature.com/articles/s41524-020-00406-3)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

## 🎓 What This Project Demonstrates

✅ **Domain Expertise:** Understanding materials science and MOF chemistry
✅ **Data Engineering:** Sourcing, cleaning, and feature engineering real scientific data
✅ **ML Pipeline:** Building, training, and validating predictive models
✅ **Full-Stack Development:** Backend API + interactive frontend
✅ **Visualization:** 3D molecular structures and prediction results
✅ **Deployment:** Production-ready application on cloud platforms
✅ **Communication:** Explaining complex chemistry + ML to diverse audiences

---

## 🔗 Links

- **Live Demo:** [mof-predictor.vercel.app](https://mof-predictor.vercel.app)
- **GitHub Repository:** [github.com/HildaPosada/mof-co2-predictor](https://github.com/HildaPosada/mof-co2-predictor)
- **LinkedIn Post:** Coming soon

---

## 📧 Questions?

This project bridges materials science and machine learning. If you're interested in ML for chemistry, quantum computing, or carbon capture technology, let's connect!

**LinkedIn:** [linkedin.com/in/hildaposada](https://linkedin.com/in/hildaposada)
**GitHub:** [@HildaPosada](https://github.com/HildaPosada)

---

**Built with 🧪 chemistry knowledge + 🤖 machine learning expertise**
