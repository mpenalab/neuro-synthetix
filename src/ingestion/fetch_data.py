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

    # 2. Búsqueda masiva sin filtros previos de potencia
    activity = new_client.activity
    # Quitamos standard_type e IC50 para ver TODO lo que hay del target
    res = activity.filter(target_chembl_id=chembl_id) 
    
    print(f"📥 Descargando datos brutos (esto puede tomar un poco más)...")
    
    activity_list = list(res)
    df = pd.DataFrame(activity_list)

    # 3. Filtrado manual agresivo en Pandas
    # Buscamos cualquier cosa que se parezca a una medida de potencia (IC50, Ki, EC50)
    potency_types = ['IC50', 'Ki', 'EC50', 'Kd']
    df_clean = df[df['standard_type'].isin(potency_types)].copy()
    
    # Limpieza de nulos críticos
    df_clean = df_clean.dropna(subset=['canonical_smiles', 'standard_value'])
    
    # Asegurar que los valores sean numéricos
    df_clean['standard_value'] = pd.to_numeric(df_clean['standard_value'], errors='coerce')
    df_clean['pchembl_value'] = pd.to_numeric(df_clean['pchembl_value'], errors='coerce')
    df_clean = df_clean.dropna(subset=['standard_value'])

    # Si pchembl_value es nulo, lo calculamos (pIC50 = -log10(IC50 * 1e-9))
    # Solo si el standard_unit es nM
    mask = df_clean['pchembl_value'].isna() & (df_clean['standard_units'] == 'nM')
    import numpy as np
    df_clean.loc[mask, 'pchembl_value'] = -np.log10(df_clean.loc[mask, 'standard_value'] * 1e-9)
    
    # Volvemos a limpiar por si quedaron NaNs en el target final
    df_clean = df_clean.dropna(subset=['pchembl_value'])

    # Guardar
    output_path = "data/raw/egfr_compounds.csv"
    os.makedirs("data/raw", exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print(f"💾 ¡Éxito! Registros brutos encontrados: {len(df)}")
    print(f"🧪 Compuestos con métricas de potencia válidas: {len(df_clean)}")

if __name__ == "__main__":
    fetch_target_data()