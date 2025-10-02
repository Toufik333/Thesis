import os
import numpy as np
import tensorflow as tf
from flwr.simulation import run_simulation
from flwr.server import ServerConfig
from server_app import app as server_app
from client_app import app as client_app
from pathlib import Path

class FederatedLearningOrchestrator:
    """
    Main orchestrator for Flower-based federated learning
    Handles breast cancer detection with non-IID datasets
    """
    
    def __init__(self, num_clients=4):
        self.num_clients = num_clients
        self.client_datasets = []
        self.config = {
            "num-server-rounds": 10,
            "local-epochs": 5,
            "batch-size": 32,
            "learning-rate": 0.001,
            "fraction-fit": 1.0,
            "fraction-evaluate": 1.0,
            "min-fit-clients": num_clients,
            "min-evaluate-clients": num_clients,
            "min-available-clients": num_clients,
            "use-fedprox": True,  # Use FedProx for non-IID data
            "proximal-mu": 0.1
        }
    
    def add_dataset(self, dataset_path: str):
        """Add a dataset path for a client"""
        if len(self.client_datasets) < self.num_clients:
            if not os.path.exists(dataset_path):
                print(f"⚠ Warning: Dataset path does not exist: {dataset_path}")
            
            self.client_datasets.append(dataset_path)
            print(f"✓ Dataset {len(self.client_datasets)} added: {dataset_path}")
        else:
            print(f"✗ Error: Maximum number of clients ({self.num_clients}) reached")
    
    def set_config(self, **kwargs):
        """
        Update configuration parameters
        
        Available parameters:
            - num_server_rounds: Number of federated learning rounds
            - local_epochs: Number of local training epochs per round
            - batch_size: Batch size for training
            - learning_rate: Learning rate for optimization
            - use_fedprox: Use FedProx (True) or FedAvg (False)
            - proximal_mu: Proximal term coefficient for FedProx
        """
        for key, value in kwargs.items():
            config_key = key.replace('_', '-')
            if config_key in self.config:
                self.config[config_key] = value
                print(f"✓ Configuration updated: {config_key} = {value}")
            else:
                print(f"⚠ Warning: Unknown configuration key: {key}")
    
    def run(self):
        """Run the federated learning simulation"""
        if len(self.client_datasets) != self.num_clients:
            print(f"\n✗ Error: Expected {self.num_clients} datasets, but got {len(self.client_datasets)}")
            print("   Use add_dataset() to add dataset paths for each client")
            return
        
        print("\n" + "="*70)
        print("🌻 FLOWER FEDERATED LEARNING - BREAST CANCER DETECTION")
        print("="*70)
        print(f"Framework: Flower (Federated Learning Framework)")
        print(f"Task: Breast Cancer Detection with Non-IID Data")
        print(f"Number of clients: {self.num_clients}")
        print(f"Strategy: {'FedProx' if self.config['use-fedprox'] else 'FedAvg'}")
        print(f"Rounds: {self.config['num-server-rounds']}")
        print(f"Local epochs: {self.config['local-epochs']}")
        print("="*70 + "\n")
        
        # Create client resource specifications
        backend_config = {
            "client_resources": {
                "num_cpus": 2,
                "num_gpus": 0.25  # Adjust based on your GPU availability
            }
        }
        
        # Create node configurations for each client
        client_app_config = []
        for i, dataset_path in enumerate(self.client_datasets):
            client_app_config.append({
                "partition-id": i,
                "dataset-path": dataset_path
            })
        
        try:
            print("🚀 Starting Flower simulation...\n")
            
            # Run the simulation
            history = run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=self.num_clients,
                backend_config=backend_config,
                client_app_config=client_app_config,
                server_app_config=self.config
            )
            
            print("\n" + "="*70)
            print("✅ FEDERATED LEARNING COMPLETE!")
            print("="*70)
            print(f"Total rounds: {len(history.losses_distributed)}")
            print(f"Final distributed loss: {history.losses_distributed[-1][1]:.4f}")
            
            if history.metrics_distributed.get("accuracy"):
                final_accuracy = history.metrics_distributed["accuracy"][-1][1]
                print(f"Final distributed accuracy: {final_accuracy:.4f}")
            
            if history.metrics_distributed.get("auc"):
                final_auc = history.metrics_distributed["auc"][-1][1]
                print(f"Final distributed AUC: {final_auc:.4f}")
            
            print("="*70 + "\n")
            
            return history
            
        except Exception as e:
            print(f"\n✗ Error during federated learning: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_global_model(self, test_dataset_path: str):
        """
        Evaluate the trained global model on test data
        
        Args:
            test_dataset_path: Path to the test dataset
        """
        print("\n" + "="*70)
        print("📊 EVALUATING GLOBAL MODEL")
        print("="*70)
        
        try:
            # Load the saved model
            model = tf.keras.models.load_model('federated_breast_cancer_model.h5')
            print("✓ Global model loaded successfully")
            
            # Load test data
            print(f"Loading test data from: {test_dataset_path}")
            test_data = np.load(test_dataset_path, allow_pickle=True)
            
            if isinstance(test_data, np.ndarray):
                X_test = test_data[:, 0]
                y_test = test_data[:, 1]
            else:
                X_test = test_data['images']
                y_test = test_data['labels']
            
            # Preprocess
            if len(X_test.shape) == 1:
                X_test = np.array([img for img in X_test])
            
            X_test = X_test.astype('float32')
            if X_test.max() > 1.0:
                X_test = X_test / 255.0
            
            if len(y_test.shape) == 1:
                y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
            
            print(f"✓ Test data loaded: {len(X_test)} samples")
            
            # Evaluate
            test_loss, test_accuracy, test_auc = model.evaluate(
                X_test, y_test, verbose=1
            )
            
            print("\n" + "-"*70)
            print("📈 TEST RESULTS:")
            print(f"  - Test Loss: {test_loss:.4f}")
            print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"  - Test AUC: {test_auc:.4f}")
            print("-"*70)
            print("="*70 + "\n")
            
            return {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_auc": test_auc
            }
            
        except Exception as e:
            print(f"✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Initialize orchestrator with 4 clients for non-IID datasets
    orchestrator = FederatedLearningOrchestrator(num_clients=4)
    
    # Configure federated learning parameters
    orchestrator.set_config(
        num_server_rounds=10,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        use_fedprox=True,  # Use FedProx for better non-IID handling
        proximal_mu=0.1    # Proximal term coefficient
    )
    
    # Add your 4 non-IID datasets
    # IMPORTANT: Replace these paths with your actual dataset paths
    orchestrator.add_dataset('path/to/client1_dataset.npy')
    orchestrator.add_dataset('path/to/client2_dataset.npy')
    orchestrator.add_dataset('path/to/client3_dataset.npy')
    orchestrator.add_dataset('path/to/client4_dataset.npy')
    
    # Run federated learning
    history = orchestrator.run()
    
    # Optional: Evaluate on test dataset after training
    if history is not None:
        # Uncomment and add path to your test dataset
        # orchestrator.evaluate_global_model('path/to/test_dataset.npy')
        pass
