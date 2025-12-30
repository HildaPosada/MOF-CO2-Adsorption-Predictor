@echo off
REM MOF CO2 Adsorption Predictor - Complete Setup Script (Windows)
REM This script automates the entire setup and training process

echo =======================================================================
echo 🔬 MOF CO2 Adsorption Predictor - Automated Setup (Windows)
echo =======================================================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Step 1: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ⚠️  Virtual environment already exists, skipping...
)
echo.

REM Step 2: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated
echo.

REM Step 3: Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt --quiet
echo ✅ Dependencies installed
echo.

REM Step 4: Verify installation
echo Verifying installation...
python test_setup.py
echo ✅ Installation verified
echo.

REM Step 5: Generate dataset
echo Generating MOF dataset...
python src/data_collection.py
echo ✅ Dataset generated
echo.

REM Step 6: Preprocess data
echo Preprocessing data and engineering features...
python src/preprocessing.py
echo ✅ Data preprocessed
echo.

REM Step 7: Train model
echo Training neural network (this may take a few minutes)...
python src/train_model.py
echo ✅ Model trained
echo.

REM Step 8: Test prediction
echo Testing predictions...
python example_prediction.py
echo ✅ Predictions tested
echo.

REM Summary
echo.
echo =======================================================================
echo 🎉 SETUP COMPLETE!
echo =======================================================================
echo.
echo ✅ All components are ready!
echo.
echo 📊 Generated Files:
echo    - data\raw\mof_co2_properties.csv
echo    - data\processed\X_train.npy, y_train.npy (and test sets)
echo    - models\mof_predictor.h5
echo    - models\scaler.pkl
echo    - models\training_history.png
echo.
echo 🚀 Next Steps:
echo.
echo 1. Start the API server:
echo    python src\api.py
echo.
echo 2. In a new terminal, start the frontend:
echo    cd frontend
echo    npm install
echo    npm start
echo.
echo 3. Visit http://localhost:3000 in your browser!
echo.
echo 📚 Documentation:
echo    - QUICKSTART.md - Quick start guide
echo    - README.md - Full documentation
echo    - WORKFLOW.md - Detailed workflow
echo    - PROJECT_SUMMARY.md - Complete overview
echo.
echo =======================================================================
pause
