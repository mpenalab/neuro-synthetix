import torch
import optuna
import mlflow
from torch_geometric.loader import DataLoader
from gnn_model import GCN
from torch_geometric.data import Data
import json
import os

# Seguridad para PyTorch 2.6+
torch.serialization.add_safe_globals([Data])

def objective(trial):
    # 1. Definir el espacio de búsqueda
    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    
    # 2. Cargar datos
    dataset = torch.load("data/processed/graphs_dataset.pt", weights_only=False)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 3. Configurar modelo y entrenamiento
    model = GCN(hidden_channels=hidden_channels, input_dim=dataset[0].num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Entrenamiento corto para cada trial (15 épocas)
    model.train()
    for epoch in range(15):
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

    # 4. Evaluación en set de validación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data.x, data.edge_index, data.batch)
            val_loss += criterion(out, data.y.view(-1, 1)).item()
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

if __name__ == "__main__":
    mlflow.set_experiment("GNN-Optimization-Optuna")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20) # Probaremos 20 combinaciones

    print("\n🏆 Mejores Hiperparámetros encontrados:")
    print(study.best_params)
    
    with mlflow.start_run(run_name="Best_Optuna_Result"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)
        
        best_params = study.best_params
        # Añadimos la mejor pérdida por registro
        best_params['best_val_loss'] = study.best_value
        
        os.makedirs("models", exist_ok=True)
        with open("models/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
        
        print("✨ Mejores parámetros guardados en models/best_params.json")