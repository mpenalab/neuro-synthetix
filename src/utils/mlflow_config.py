import mlflow
import os

def setup_mlflow(experiment_name="Neuro-Synthetix-GNN"):
    """Configura el servidor de tracking de MLflow local."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    print(f"🧪 MLflow conectado al experimento: {experiment_name}")

if __name__ == "__main__":
    # Esto servirá para probar la conexión
    setup_mlflow()