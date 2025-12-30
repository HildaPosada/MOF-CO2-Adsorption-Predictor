"""
MOF CO2 Adsorption Dataset Generator

This script generates a synthetic dataset of Metal-Organic Framework (MOF) properties
and their corresponding CO2 adsorption capacities for machine learning model training.

The dataset includes realistic MOF structural properties based on CoRE MOF database statistics.
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_mof_dataset(n_samples=40):
    """
    Generate synthetic MOF dataset with realistic property distributions.
    
    Parameters:
    -----------
    n_samples : int
        Number of MOF structures to generate (default: 40)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing MOF properties and CO2 adsorption values
    """
    
    print(f"🔬 Generating {n_samples} MOF structures...")
    
    # MOF naming convention
    mof_names = [f"MOF-{i:03d}" for i in range(1, n_samples + 1)]
    
    # Structural properties (based on CoRE MOF database statistics)
    # Surface area: 500-6000 m²/g (typical range for MOFs)
    surface_area = np.random.uniform(500, 6000, n_samples)
    
    # Pore volume: 0.2-2.5 cm³/g
    pore_volume = np.random.uniform(0.2, 2.5, n_samples)
    
    # Framework density: 0.2-2.0 g/cm³
    framework_density = np.random.uniform(0.2, 2.0, n_samples)
    
    # Pore diameter: 3-30 Å
    pore_diameter = np.random.uniform(3, 30, n_samples)
    
    # Void fraction: 0.3-0.9
    void_fraction = np.random.uniform(0.3, 0.9, n_samples)
    
    # Largest cavity diameter: 5-40 Å
    largest_cavity = np.random.uniform(5, 40, n_samples)
    
    # Pore limiting diameter: 2-20 Å
    pore_limiting_diameter = np.random.uniform(2, 20, n_samples)
    
    # Chemical composition features
    # Metal content: 5-40% by weight
    metal_content = np.random.uniform(5, 40, n_samples)
    
    # Organic linker weight: 60-95%
    organic_content = 100 - metal_content
    
    # Functional groups (binary features)
    has_amino_group = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    has_carboxylate = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    has_hydroxyl = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Metal types (one-hot encoded simplified)
    metal_types = np.random.choice(['Zn', 'Cu', 'Zr', 'Al', 'Cr'], n_samples)
    
    # Generate CO2 adsorption capacity (mol/kg at 298K, 1 bar)
    # Based on correlation with structural properties
    co2_adsorption = (
        0.0008 * surface_area +  # Surface area contribution
        1.5 * pore_volume +      # Pore volume contribution
        -0.5 * framework_density + # Density (inverse correlation)
        0.05 * pore_diameter +   # Pore size contribution
        2.0 * void_fraction +    # Void fraction contribution
        0.3 * has_amino_group +  # Amino group enhancement
        np.random.normal(0, 0.3, n_samples)  # Random noise
    )
    
    # Clip to realistic range: 0.5-7.5 mol/kg
    co2_adsorption = np.clip(co2_adsorption, 0.5, 7.5)
    
    # Create DataFrame
    data = pd.DataFrame({
        'MOF_Name': mof_names,
        'Surface_Area': surface_area,
        'Pore_Volume': pore_volume,
        'Framework_Density': framework_density,
        'Pore_Diameter': pore_diameter,
        'Void_Fraction': void_fraction,
        'Largest_Cavity_Diameter': largest_cavity,
        'Pore_Limiting_Diameter': pore_limiting_diameter,
        'Metal_Content': metal_content,
        'Organic_Content': organic_content,
        'Has_Amino_Group': has_amino_group,
        'Has_Carboxylate': has_carboxylate,
        'Has_Hydroxyl': has_hydroxyl,
        'Metal_Type': metal_types,
        'CO2_Adsorption_mol_kg': co2_adsorption
    })
    
    return data

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("MOF CO2 Adsorption Dataset Generator")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate dataset
    df = generate_mof_dataset(n_samples=40)
    
    # Save to CSV
    output_path = 'data/raw/mof_co2_properties.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Total MOF structures: {len(df)}")
    print(f"   - Features: {len(df.columns) - 2}")  # Exclude name and target
    print(f"   - CO2 adsorption range: {df['CO2_Adsorption_mol_kg'].min():.2f} - {df['CO2_Adsorption_mol_kg'].max():.2f} mol/kg")
    print(f"   - Surface area range: {df['Surface_Area'].min():.0f} - {df['Surface_Area'].max():.0f} m²/g")
    print(f"   - Pore volume range: {df['Pore_Volume'].min():.2f} - {df['Pore_Volume'].max():.2f} cm³/g")
    
    print("\n🔍 Preview of first 5 MOF structures:")
    print(df.head())
    
    print("\n✨ Ready for preprocessing! Run: python src/preprocessing.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
