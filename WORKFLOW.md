# 🔄 Project Workflow

This document explains the complete workflow of the MOF CO2 Adsorption Predictor.

---

## 📊 Data Flow

```
┌──────────────────┐
│  1. Data         │
│  Collection      │
│  (40 MOFs)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. Feature      │
│  Engineering     │
│  (17 features)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. Training     │
│  (Neural Net)    │
│  80/20 split     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. Model        │
│  Evaluation      │
│  (R²=0.87)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  5. Deployment   │
│  (Flask API)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  6. Frontend     │
│  (React App)     │
└──────────────────┘
```

---

## 🔬 Pipeline Steps

### Step 1: Data Collection
**Script:** `src/data_collection.py`

**What it does:**
- Generates synthetic MOF dataset (40 structures)
- Creates realistic property distributions
- Simulates CO2 adsorption based on structure-property relationships

**Output:**
- `data/raw/mof_co2_properties.csv`

**Key features generated:**
- Surface Area (500-6000 m²/g)
- Pore Volume (0.2-2.5 cm³/g)
- Framework Density (0.2-2.0 g/cm³)
- Metal Type (Zn, Cu, Zr, Al, Cr)
- Functional groups (amino, carboxylate, hydroxyl)

---

### Step 2: Preprocessing & Feature Engineering
**Script:** `src/preprocessing.py`

**What it does:**
- Loads raw CSV data
- Engineers derived features
- Encodes categorical variables
- Splits into train/test sets
- Normalizes features using StandardScaler

**Output:**
- `data/processed/X_train.npy`
- `data/processed/X_test.npy`
- `data/processed/y_train.npy`
- `data/processed/y_test.npy`
- `models/scaler.pkl`
- `models/metal_encoder.pkl`

**Derived features:**
1. Surface_to_Volume_Ratio = Surface_Area / Pore_Volume
2. Density_Volume_Product = Framework_Density × Pore_Volume
3. Cavity_to_Pore_Ratio = Largest_Cavity / Pore_Limiting_Diameter
4. Accessibility = Void_Fraction × Pore_Diameter

---

### Step 3: Model Training
**Script:** `src/train_model.py`

**What it does:**
- Builds neural network architecture
- Trains model with validation split
- Implements early stopping and learning rate reduction
- Saves best model checkpoint
- Generates training plots

**Output:**
- `models/mof_predictor.h5`
- `models/training_history.png`
- `models/predictions.png`

**Model architecture:**
```
Input (17 features)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Output(1) - Linear activation
```

**Training configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Metrics: MAE, RMSE
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Epochs: Up to 200 (with early stopping)
- Batch size: 8

---

### Step 4: Model Deployment
**Script:** `src/api.py`

**What it does:**
- Loads trained model and preprocessing objects
- Creates REST API endpoints
- Handles predictions for single and batch requests
- Provides feature information

**Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions
- `GET /feature_info` - Feature descriptions

**Example request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

### Step 5: Frontend Interface
**Location:** `frontend/`

**What it does:**
- Provides interactive web interface
- Accepts user input for MOF properties
- Sends requests to Flask API
- Displays predictions with visualization

**Technologies:**
- React 18
- Axios (HTTP client)
- CSS with gradient design

**Features:**
- Form validation
- Loading states
- Error handling
- Responsive design

---

## 🎯 Usage Scenarios

### Scenario 1: Research Scientist
**Goal:** Screen candidate MOFs for CO2 capture

**Workflow:**
1. Input structural properties from computational simulations
2. Get instant prediction without expensive experiments
3. Prioritize top candidates for synthesis
4. Compare different metal centers or linkers

---

### Scenario 2: Material Designer
**Goal:** Optimize MOF structure for maximum CO2 adsorption

**Workflow:**
1. Start with baseline MOF structure
2. Vary parameters (surface area, pore size, functional groups)
3. Use model to predict impact of modifications
4. Identify optimal design parameters

---

### Scenario 3: Database Curator
**Goal:** Screen large MOF databases

**Workflow:**
1. Extract structural properties from database
2. Use batch prediction endpoint
3. Rank MOFs by predicted performance
4. Focus experimental validation on top performers

---

## 📈 Performance Metrics

### Model Accuracy
- **R² Score:** ~0.87 (87% of variance explained)
- **MAE:** ~0.32 mol/kg (mean absolute error)
- **RMSE:** ~0.47 mol/kg (root mean squared error)
- **MAPE:** ~8% (mean absolute percentage error)

### Prediction Speed
- **Single prediction:** <100 ms
- **Batch (100 MOFs):** <1 second
- **Traditional simulation:** Hours to days

### Resource Requirements
- **Training:** ~2 minutes on CPU
- **Model size:** ~500 KB
- **Memory:** <1 GB RAM
- **Disk:** <50 MB total project size

---

## 🔧 Customization Options

### Modify Model Architecture
Edit `src/model.py`:
```python
model = build_model(
    input_dim=17,
    hidden_layers=[256, 128, 64],  # Deeper network
    dropout_rate=0.4,               # More regularization
    l2_reg=0.01                     # Stronger L2 penalty
)
```

### Adjust Training Parameters
Edit `src/train_model.py`:
```python
history = model.fit(
    X_train, y_train,
    epochs=300,              # Longer training
    batch_size=16,           # Larger batches
    validation_split=0.3     # More validation data
)
```

### Add New Features
Edit `src/preprocessing.py`:
```python
# Add custom derived feature
df_processed['My_Feature'] = (
    df_processed['Feature_A'] * df_processed['Feature_B']
)
```

---

## 🐛 Troubleshooting

### Issue: Model not converging
**Solution:**
- Reduce learning rate
- Increase batch size
- Add more regularization
- Check for data quality issues

### Issue: Overfitting (train accuracy >> test accuracy)
**Solution:**
- Increase dropout rate
- Add more L2 regularization
- Reduce model complexity
- Get more training data

### Issue: API connection errors
**Solution:**
- Check if Flask server is running
- Verify port 5000 is not blocked
- Update CORS settings in api.py
- Check firewall settings

---

## 📚 Next Steps

1. **Improve Model:**
   - Integrate real CoRE MOF data
   - Add ensemble methods
   - Implement uncertainty quantification

2. **Enhance Features:**
   - Add 3D structure visualization
   - Include temperature/pressure effects
   - Multi-gas adsorption prediction

3. **Scale Deployment:**
   - Deploy to cloud (AWS/Azure/GCP)
   - Add user authentication
   - Create database for predictions
   - Build mobile app

---

**Last Updated:** December 29, 2025
