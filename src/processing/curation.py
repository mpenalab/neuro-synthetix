import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from tqdm import tqdm
import os

def curate_molecules(input_path="data/raw/egfr_compounds.csv"):
    print(f"🧪 Iniciando curación química de {input_path}...")
    
    df = pd.read_csv(input_path)
    initial_count = len(df)
    
    # 1. Inicializar el removedor de sales de RDKit
    remover = SaltRemover.SaltRemover()
    
    valid_data = []
    
    for _, row in tqdm(df.iterrows(), total=initial_count, desc="Curando SMILES"):
        smiles = row['canonical_smiles']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            # 2. Remover sales (ej. HCl, NaOH que vienen con la molécula)
            mol = remover.StripMol(mol)
            
            # 3. Obtener el SMILES canónico (único para cada estructura)
            clean_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            
            valid_data.append({
                'molecule_chembl_id': row['molecule_chembl_id'],
                'smiles': clean_smiles,
                'pchembl_value': row['pchembl_value'],
                'standard_value': row['standard_value']
            })
    
    # 4. Crear DataFrame y eliminar duplicados estructurales
    df_curated = pd.DataFrame(valid_data)
    df_curated = df_curated.drop_duplicates(subset=['smiles'])
    
    # 5. Guardar el dataset procesado
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "egfr_curated.csv")
    
    df_curated.to_csv(output_path, index=False)
    
    print(f"\n✨ Proceso completado:")
    print(f"   - Moléculas iniciales: {initial_count}")
    print(f"   - Moléculas curadas: {len(df_curated)}")
    print(f"   - Eliminadas (ruido/duplicados): {initial_count - len(df_curated)}")
    print(f"📂 Archivo guardado en: {output_path}")

if __name__ == "__main__":
    curate_molecules()