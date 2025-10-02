# server.py

from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# --- 1. Model Definition ---
# NOTE: In a real-world application, it's best practice to define the model
# in a separate file (e.g., model.py) and import it in both client.py and
# server.py. For this self-contained example, we redefine it here.
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron model, identical to the client's."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Custom FedProx Strategy ---
# We implement FedProx by creating a custom strategy that inherits from FedAvg.
# The core idea of FedProx is handled on the client side (see client.py).
# The server's role is to simply orchestrate the rounds and, importantly,
# pass the `mu` hyperparameter to the clients via the configuration function.
class FedProx(FedAvg):
    """
    Custom FedProx strategy.
    
    This strategy is a simple extension of FedAvg. It doesn't change the
    aggregation logic but is used to demonstrate how to pass custom
    configurations (like the FedProx `mu` parameter) to clients.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        # This is the standard FedAvg aggregation. FedProx does not change
        # the server-side aggregation, only the client-side loss function.
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        return aggregated_parameters, aggregated_metrics

# --- 3. Server Configuration Logic ---
def fit_config(server_round: int) -> Dict[str, Scalar]:
    """
    Return training configuration dict for each round.
    
    This function is sent to the FedProx strategy and is called by the server
    at the beginning of each training round. It's used to configure the
    clients for the `fit` method.
    
    Args:
        server_round (int): The current round of federated learning.
        
    Returns:
        Dict[str, Scalar]: A dictionary containing the parameters for clients.
    """
    config = {
        "server_round": server_round,
        "local_epochs": 2,  # Number of local epochs for clients
        "mu": 0.01,         # The FedProx proximal term coefficient
    }
    return config

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # Define the total number of clients and rounds
    NUM_CLIENTS = 4
    NUM_ROUNDS = 10

    # Instantiate the model to get its initial parameters
    # This is needed to initialize the global model on the server
    model = MLP(input_size=30, hidden_size=64, output_size=2)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Define the FedProx strategy
    # The strategy orchestrates the federated learning process
    strategy = FedProx(
        # The fraction of clients to use for training in each round
        fraction_fit=1.0,
        # The fraction of clients to use for evaluation
        fraction_evaluate=1.0,
        # The minimum number of clients to be available for training
        min_fit_clients=NUM_CLIENTS,
        # The minimum number of clients to be available for evaluation
        min_evaluate_clients=NUM_CLIENTS,
        # The minimum number of available clients required to start a round
        min_available_clients=NUM_CLIENTS,
        # A function that configures client training for each round
        on_fit_config_fn=fit_config,
        # The initial parameters of the global model
        initial_parameters=ndarrays_to_parameters(model_parameters),
    )

    print("=" * 80)
    print("Flower Server")
    print(f"Strategy: FedProx")
    print(f"Total Rounds: {NUM_ROUNDS}")
    print(f"Waiting for {NUM_CLIENTS} clients to connect...")
    print("=" * 80)

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )