#!/usr/bin/env python
"""
Quick test script to verify the MOF CO2 Predictor setup.

This script tests the data pipeline and model prediction workflow.
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def test_imports():
    """Test if all required Python packages are installed"""
    print("\n" + "=" * 60)
    print("Testing Python Dependencies")
    print("=" * 60)
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('tensorflow', 'TensorFlow'),
        ('flask', 'Flask'),
        ('joblib', 'Joblib')
    ]
    
    all_imported = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} NOT INSTALLED")
            all_imported = False
    
    return all_imported

def test_project_structure():
    """Test if project structure is correct"""
    print("\n" + "=" * 60)
    print("Testing Project Structure")
    print("=" * 60)
    
    required_files = [
        ('src/data_collection.py', 'Data collection script'),
        ('src/preprocessing.py', 'Preprocessing script'),
        ('src/model.py', 'Model architecture'),
        ('src/train_model.py', 'Training script'),
        ('src/api.py', 'Flask API'),
        ('requirements.txt', 'Requirements file'),
        ('QUICKSTART.md', 'Quick start guide'),
        ('frontend/package.json', 'Frontend package.json'),
        ('frontend/src/App.js', 'Frontend App.js')
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if not check_file(filepath, description):
            all_exist = False
    
    return all_exist

def test_data_generation():
    """Test data generation"""
    print("\n" + "=" * 60)
    print("Testing Data Generation")
    print("=" * 60)
    
    try:
        # Add src to path if not already there
        src_path = os.path.join(os.getcwd(), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Import and run data collection
        from data_collection import generate_mof_dataset
        
        df = generate_mof_dataset(n_samples=10)
        
        if len(df) == 10:
            print(f"✅ Generated {len(df)} MOF structures")
            print(f"✅ Dataset shape: {df.shape}")
            print(f"✅ Features: {list(df.columns)}")
            return True
        else:
            print(f"❌ Expected 10 samples, got {len(df)}")
            return False
    
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print(f"   Make sure you're running this script from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 MOF CO2 Predictor - Setup Verification")
    print("=" * 60)
    
    # Run tests
    test1 = test_imports()
    test2 = test_project_structure()
    test3 = test_data_generation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    results = {
        'Dependencies': test1,
        'Project Structure': test2,
        'Data Generation': test3
    }
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Generate full dataset: python src/data_collection.py")
        print("2. Preprocess data: python src/preprocessing.py")
        print("3. Train model: python src/train_model.py")
        print("4. Start API: python src/api.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
