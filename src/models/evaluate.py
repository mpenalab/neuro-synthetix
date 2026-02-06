import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from gnn_model import GCN
from torch_geometric.data import Data
import os
import json

# Seguridad para PyTorch 2.6+
torch.serialization.add_safe_globals([Data])

def evaluate():
    print("🧪 Iniciando evaluación del modelo...")
    # 1. Cargar datos y modelo
    dataset = torch.load("data/processed/graphs_dataset.pt", weights_only=False)
    
    # Usamos el mismo split que en la optimización para ser justos
    train_size = int(0.8 * len(dataset))
    test_dataset = dataset[train_size:]
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Cargar hiperparámetros y pesos
    with open("models/best_params.json", "r") as f:
        params = json.load(f)
    
    model = GCN(hidden_channels=params['hidden_channels'], input_dim=dataset[0].num_node_features)
    model.load_state_dict(torch.load("models/gnn_v1.pth"))
    model.eval()

    y_true, y_pred = [], []

    # 2. Predicción
    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            y_true.append(data.y.item())
            y_pred.append(out.item())

    # 3. Cálculo de métricas
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"📈 Métricas finales en Test Set:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - R²:  {r2:.4f}")

    # 4. Gráfico de Residuos (Actual vs Predicted)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--r', lw=2)
    plt.xlabel("Valor Real (pChEMBL)")
    plt.ylabel("Predicción (pChEMBL)")
    plt.title(f"Evaluación del Modelo (R²: {r2:.2f})")
    
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/evaluation_plot.png")
    print("📊 Gráfico guardado en reports/figures/evaluation_plot.png")

if __name__ == "__main__":
    evaluate()