import torch
import json
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from captum.attr import IntegratedGradients
from torch_geometric.data import Data

# Importar tu arquitectura para que PyTorch pueda cargar el modelo
from gnn_model import GCN 

# Seguridad para PyTorch 2.6+
torch.serialization.add_safe_globals([Data])

def generate_heatmaps():
    print("🎨 Iniciando generación de mapas de calor químicos...")
    
    # 1. Configuración de rutas robusta
    base_path = Path(__file__).resolve().parent.parent.parent
    processed_data_path = base_path / "data" / "processed" / "graphs_dataset.pt"
    params_path = base_path / "models" / "best_params.json"
    model_weights_path = base_path / "models" / "gnn_v1.pth"
    output_dir = base_path / "reports" / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "molecule_explanation.png"

    # 2. Carga de artefactos
    if not processed_data_path.exists():
        print(f"❌ Error: No se encuentra el dataset en {processed_data_path}")
        return

    dataset = torch.load(processed_data_path, weights_only=False)
    
    with open(params_path, "r") as f:
        params = json.load(f)
    
    # Instanciar modelo con los mejores hiperparámetros encontrados
    model = GCN(hidden_channels=params['hidden_channels'], input_dim=dataset[0].num_node_features)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # 3. Selección de molécula y cálculo de atribuciones
    # Usamos la primera molécula del dataset (puedes cambiar el índice)
    data = dataset[0]
    smiles = getattr(data, 'smiles', None)
    
    if smiles is None:
        print("❌ Error: El objeto Data no contiene el atributo 'smiles'. Re-ejecuta graph_construction.py.")
        return

    mol = Chem.MolFromSmiles(smiles)
    
    def model_forward(x, edge_index, batch):
        return model(x, edge_index, batch)

    ig = IntegratedGradients(model_forward)
    data.x.requires_grad = True
    
    # Atribución con internal_batch_size=1 para evitar errores de dimensiones en GNN
    attributions = ig.attribute(
        data.x.unsqueeze(0), 
        target=0, 
        additional_forward_args=(data.edge_index, data.batch),
        internal_batch_size=1
    )
    
    # Sumar atribuciones por átomo y convertir a lista para RDKit
    weights = attributions.squeeze(0).sum(dim=1).detach().numpy().tolist()

    # 4. Renderizado con RDKit (Cairo)
    try:
        # Generar el canvas de dibujo
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        
        # Crear el mapa de similitud basado en los pesos de Captum
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, 
            weights, 
            draw2d=drawer,
            contourLines=10,
            alpha=0.3
        )
        drawer.FinishDrawing()

        # Guardar el archivo físico
        with open(output_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        print(f"✅ ¡Éxito! Mapa de calor guardado en: {output_path}")
        
    except Exception as e:
        print(f"❌ Error durante el renderizado químico: {e}")

if __name__ == "__main__":
    generate_heatmaps()