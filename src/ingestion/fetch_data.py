import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import os

def fetch_target_data(target_hgnc_symbol="EGFR"):
    """
    Extrae datos de bioactividad para una proteína específica desde ChEMBL.
    Filtra por ensayos de unión (Binding) y medidas de IC50 en unidades nM.
    """
    print(f"🚀 Iniciando búsqueda para el target: {target_hgnc_symbol}...")
    
    # 1. Buscar el ID de la proteína (Target)
    target = new_client.target
    target_query = target.search(target_hgnc_symbol)
    targets = pd.DataFrame.from_dict(target_query)
    
    # Seleccionamos el primer resultado (el más relevante)
    chembl_id = targets.loc[0, 'target_chembl_id']
    print(f"✅ Target hallado: {chembl_id}")

    # 2. Filtrar actividades para este target
    # Filtramos por: IC50 (potencia), ensayos tipo 'B' (Binding) y solo datos en nM
    activity = new_client.activity
    res = activity.filter(target_chembl_id=chembl_id) \
                  .filter(standard_type="IC50") \
                  .filter(standard_units="nM")

    print(f"📥 Descargando datos de bioactividad (esto puede tardar unos minutos)...")
    
    # Convertimos a DataFrame
    df = pd.DataFrame.from_dict(res)
    
    # 3. Selección de columnas críticas para MLOps
    columns_to_keep = [
        'molecule_chembl_id', 
        'canonical_smiles', 
        'standard_value', 
        'standard_units',
        'pchembl_value'
    ]
    
    df_clean = df[columns_to_keep].dropna(subset=['canonical_smiles', 'standard_value'])
    
    # Guardar en la carpeta de datos crudos
    output_path = "data/raw/egfr_compounds.csv"
    os.makedirs("data/raw", exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print(f"💾 ¡Éxito! Se guardaron {len(df_clean)} compuestos en: {output_path}")

if __name__ == "__main__":
    fetch_target_data()