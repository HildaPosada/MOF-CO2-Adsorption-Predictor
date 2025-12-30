#!/bin/bash

# MOF CO2 Adsorption Predictor - Complete Setup Script
# This script automates the entire setup and training process

set -e  # Exit on error

echo "======================================================================="
echo "🔬 MOF CO2 Adsorption Predictor - Automated Setup"
echo "======================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
print_success "Python version: $python_version"

# Step 1: Create virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists, skipping..."
fi

# Step 2: Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 3: Install dependencies
print_info "Installing Python dependencies..."
pip install -r requirements.txt --quiet
print_success "Dependencies installed"

# Step 4: Verify installation
print_info "Verifying installation..."
python test_setup.py
print_success "Installation verified"

# Step 5: Generate dataset
print_info "Generating MOF dataset..."
python src/data_collection.py
print_success "Dataset generated"

# Step 6: Preprocess data
print_info "Preprocessing data and engineering features..."
python src/preprocessing.py
print_success "Data preprocessed"

# Step 7: Train model
print_info "Training neural network (this may take a few minutes)..."
python src/train_model.py
print_success "Model trained"

# Step 8: Test prediction
print_info "Testing predictions..."
python example_prediction.py
print_success "Predictions tested"

# Summary
echo ""
echo "======================================================================="
echo "🎉 SETUP COMPLETE!"
echo "======================================================================="
echo ""
print_success "All components are ready!"
echo ""
echo "📊 Generated Files:"
echo "   - data/raw/mof_co2_properties.csv"
echo "   - data/processed/X_train.npy, y_train.npy (and test sets)"
echo "   - models/mof_predictor.h5"
echo "   - models/scaler.pkl"
echo "   - models/training_history.png"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Start the API server:"
echo "   python src/api.py"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   cd frontend"
echo "   npm install"
echo "   npm start"
echo ""
echo "3. Visit http://localhost:3000 in your browser!"
echo ""
echo "📚 Documentation:"
echo "   - QUICKSTART.md - Quick start guide"
echo "   - README.md - Full documentation"
echo "   - WORKFLOW.md - Detailed workflow"
echo "   - PROJECT_SUMMARY.md - Complete overview"
echo ""
echo "======================================================================="
