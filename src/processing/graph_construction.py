import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np
import os

def get_atom_features(atom):
    """Convierte un átomo de RDKit en un vector numérico (One-Hot)."""
    # Lista de átomos comunes en nuestro dataset
    allowed_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    symbol = atom.GetSymbol()
    
    # One-hot encoding del símbolo del átomo
    symbol_feat = [1 if symbol == s else 0 for s in allowed_atoms]
    # Otras propiedades químicas
    extra_feats = [
        atom.GetDegree(),            # Grado de enlace
        atom.GetImplicitValence(),   # Valencia
        1 if atom.GetIsAromatic() else 0, # ¿Es aromático?
        atom.GetFormalCharge()       # Carga
    ]
    return np.array(symbol_feat + extra_feats, dtype=np.float32)

def smiles_to_graph(smiles, target_value):
    """Convierte un SMILES en un objeto Data de PyTorch Geometric."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # 1. Obtener características de los nodos (N átomos x F características)
    node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(node_feats), dtype=torch.float)
    
    # 2. Obtener conectividad (Edge Index: 2 x M enlaces)
    edge_indices = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([start, end])
        edge_indices.append([end, start]) # Grafos no dirigidos
        
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # 3. Target (valor pChEMBL)
    y = torch.tensor([target_value], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

def process_all_graphs():
    print("🧪 Transformando moléculas en grafos...")
    df = pd.read_csv("data/processed/egfr_with_features.csv")
    
    graphs = []
    for _, row in df.iterrows():
        graph = smiles_to_graph(row['smiles'], row['pchembl_value'])
        if graph:
            graphs.append(graph)
            
    # Guardar el dataset de grafos procesado
    os.makedirs("data/processed", exist_ok=True)
    torch.save(graphs, "data/processed/graphs_dataset.pt")
    print(f"✅ Se han procesado y guardado {len(graphs)} grafos en data/processed/graphs_dataset.pt")

if __name__ == "__main__":
    process_all_graphs()