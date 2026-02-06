import torch
from captum.attr import IntegratedGradients
from gnn_model import GCN
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Draw
import json

# Configuración de seguridad
torch.serialization.add_safe_globals([Data])

def explain_molecule():
    # 1. Cargar datos y modelo
    dataset = torch.load("data/processed/graphs_dataset.pt", weights_only=False)
    with open("models/best_params.json", "r") as f:
        params = json.load(f)
    
    model = GCN(hidden_channels=params['hidden_channels'], input_dim=dataset[0].num_node_features)
    model.load_state_dict(torch.load("models/gnn_v1.pth"))
    model.eval()

    # 2. Seleccionar una molécula de ejemplo (la primera del dataset)
    data = dataset[0]
    
    # Definir una función auxiliar para Captum
    def model_forward(x, edge_index, batch):
        return model(x, edge_index, batch)

    # 3. Aplicar Integrated Gradients
    ig = IntegratedGradients(model_forward)
    
    # Necesitamos que los gradientes fluyan respecto a las características de los átomos (x)
    data.x.requires_grad = True
    # El 'target=0' es porque la salida es un único valor (pChEMBL)
    attributions = ig.attribute(
    data.x.unsqueeze(0), # Añadimos una dimensión de batch para Captum
    target=0,
    additional_forward_args=(data.edge_index, data.batch),
    internal_batch_size=1  # <--- Esto evita el error de tamaños de tensor
    )
    
    attributions = attributions.squeeze(0)

    # Sumar las atribuciones por átomo
    importance = attributions.sum(dim=1).detach().numpy()
    print(f"✅ Atribuciones calculadas para {data.num_nodes} átomos.")
    
    # 4. (Opcional) Guardar reporte de importancia
    # Mañana aprenderemos a mapear esto de vuelta a una imagen de RDKit
    return importance

if __name__ == "__main__":
    explain_molecule()