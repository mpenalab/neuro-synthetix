import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import os

def calculate_lipinski(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol)
    }

def run_eda(input_path="data/processed/egfr_curated.csv"):
    print(f"📊 Iniciando Análisis Exploratorio Científico...")
    df = pd.read_csv(input_path)
    
    # Calcular propiedades de Lipinski
    stats = df['smiles'].apply(calculate_lipinski).apply(pd.Series)
    df = pd.concat([df, stats], axis=1)
    
    # Crear carpeta para reportes
    os.makedirs("reports/figures", exist_ok=True)
    
    # Generar visualizaciones
    plt.figure(figsize=(12, 8))
    
    # 1. Distribución de Peso Molecular
    plt.subplot(2, 2, 1)
    sns.histplot(df['MW'], kde=True, color='blue')
    plt.title('Distribución de Peso Molecular (MW)')
    
    # 2. LogP (Lipofilicidad)
    plt.subplot(2, 2, 2)
    sns.histplot(df['LogP'], kde=True, color='green')
    plt.title('Distribución de LogP')
    
    # 3. pChEMBL Value (Afinidad Real)
    plt.subplot(2, 2, 3)
    sns.histplot(df['pchembl_value'], kde=True, color='red')
    plt.title('Distribución de Afinidad (pChEMBL)')
    
    # 4. Correlación MW vs LogP
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='MW', y='LogP', data=df, alpha=0.5)
    plt.title('MW vs LogP')
    
    plt.tight_layout()
    plt.savefig("reports/figures/eda_results.png")
    
    # Guardar dataset con features iniciales
    df.to_csv("data/processed/egfr_with_features.csv", index=False)
    print(f"✨ EDA completado. Reporte guardado en reports/figures/eda_results.png")

if __name__ == "__main__":
    run_eda()