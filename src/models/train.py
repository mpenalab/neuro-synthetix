import torch
from torch_geometric.loader import DataLoader
from gnn_model import GCN
import mlflow
import os
from torch_geometric.data import Data

# Configuración de seguridad para PyTorch 2.6+
torch.serialization.add_safe_globals([Data])

def train():
    # 1. Cargar datos
    dataset = torch.load("data/processed/graphs_dataset.pt", weights_only=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Inicializar modelo con LR más estable (0.001)
    model = GCN(hidden_channels=16, input_dim=dataset[0].num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    criterion = torch.nn.MSELoss()

    mlflow.set_experiment("Neuro-Synthetix-GNN")
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "GCN")
        mlflow.log_param("lr", 0.001)

        for epoch in range(1, 101):
            model.train()
            total_loss = 0
            for data in loader:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.view(-1, 1)) + (torch.randn(1).to(out.device) * 1e-6)
                
                # --- PROTECCIÓN ANTI-NAN ---
                if not torch.isnan(loss):
                    loss.backward()
                    # Clip de gradientes para estabilidad
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Loss: {avg_loss:.4f}')
                mlflow.log_metric("mse_loss", avg_loss, step=epoch)

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/gnn_v1.pth")
        mlflow.log_artifact("models/gnn_v1.pth")
        print("✅ Entrenamiento completado con éxito.")

if __name__ == "__main__":
    train()