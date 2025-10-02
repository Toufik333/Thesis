#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
server.py

Flower Federated Learning server coordinating non-IID breast cancer detection across 4 clients.
Chosen strategy: FedProx (server-side proximal regularization support) to improve robustness under non-IID data. [web:9][web:21]

Key features:
- Strategy: FedProx with fraction_fit=1.0 (select all clients each round), 10 rounds total. [web:9][web:24][web:28]
- Dynamic fit_config(server_round): Sends local training hyperparameters (epochs, batch size, proximal mu) to clients. [web:9][web:29]
- Model import: Reuses the same model architecture as in client.py for optional server-side initialization/evaluation. [web:4][web:24]
- Execution: Starts Flower server on a specified address/port using start_server and ServerConfig. [web:24]

Notes:
- start_server supports strategy, num_rounds, and server_address. Newer deployments may prefer SuperLink, but start_server remains usable. [web:24]
- FedProx constructor takes proximal_mu; clients should also receive mu in fit config for the local proximal term. [web:9]
"""

import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

import flwr as fl

# Import the same model definitions used by clients.
# Assumes client.py is in the same directory and exposes MLPBinary and SimpleCNN.
# If placed in a package/module, adjust import accordingly. [web:4]
from client import MLPBinary, SimpleCNN  # noqa: F401  # [web:4]


# -------------------------------
# Optional: Server-side evaluate
# -------------------------------
def get_server_model(model_type: str = "mlp") -> nn.Module:
    """
    Initialize a server-side model mirroring the client model architecture for optional
    centralized evaluation or parameter initialization. [web:4]
    """
    if model_type == "cnn":
        # 1x6x5 layout to mirror client-side CNN if chosen. [web:4]
        return SimpleCNN(in_channels=1, img_h=6, img_w=5)
    return MLPBinary(in_features=30, hidden=64)  # default MLP for tabular features. [web:4]


def get_evaluate_fn(model_type: str = "mlp"):
    """
    Create a server-side evaluation function compatible with Flower strategies.
    This dummy evaluation computes a constant loss without real data, serving as a placeholder.
    Replace with real centralized validation data if available. [web:24][web:29]
    """
    model = get_server_model(model_type=model_type)

    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config: Dict):
        # Load parameters into the server model for centralized evaluation. [web:24]
        state_dict = model.state_dict()
        new_state = {}
        for (k, v), p in zip(state_dict.items(), parameters):
            new_state[k] = torch.tensor(p, dtype=v.dtype)
        model.load_state_dict(new_state, strict=True)

        # Placeholder loss/metrics
        loss = float(0.0 + 0.0 * server_round)
        metrics = {"server_round": int(server_round)}
        return loss, metrics  # Flower expects (loss, metrics). [web:24]

    return evaluate_fn  # [web:24]


# -------------------------------
# Dynamic Fit Config
# -------------------------------
def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """
    Provide per-round configuration to clients:
    - local_epochs: increase epochs as rounds progress to encourage convergence.
    - batch_size: fixed here but can be adapted per round.
    - proximal_mu: FedProx proximal term (must match the server FedProx setup).
    - Optional learning rate or others can be added. [web:9][web:27][web:29]
    """
    # Simple schedule: 1 epoch for first 3 rounds, then 2, then 3+ later. [web:27]
    if server_round <= 3:
        local_epochs = 1
    elif server_round <= 6:
        local_epochs = 2
    else:
        local_epochs = 3

    config: Dict[str, fl.common.Scalar] = {
        "server_round": int(server_round),
        "local_epochs": int(local_epochs),
        "batch_size": int(32),
        "proximal_mu": float(0.01),  # aligns with FedProx server-side mu. [web:9]
    }
    return config  # [web:9][web:29]


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="[::]:8080", help="Server bind address (host:port)")  # [web:24]
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds")  # [web:24]
    parser.add_argument("--strategy", type=str, default="fedprox", choices=["fedprox", "fedadagrad"], help="FL strategy")  # [web:9][web:25]
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"], help="Model type (for eval/init)")  # [web:4]
    args = parser.parse_args()

    # Optional: initialize server-side model parameters to seed the strategy
    server_model = get_server_model(model_type=args.model)
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.detach().cpu().numpy() for _, val in server_model.state_dict().items()]
    )  # [web:24][web:29]

    # Strategy selection
    if args.strategy.lower() == "fedprox":
        # FedProx with all 4 clients selected per round (fraction_fit=1.0) and on_fit_config_fn for dynamic configs. [web:9]
        strategy = fl.server.strategy.FedProx(
            fraction_fit=1.0,             # select all available clients each round [web:9][web:28]
            fraction_evaluate=1.0,        # evaluate on all clients if evaluate is used [web:9]
            min_fit_clients=4,            # expect all 4 clients [web:9]
            min_evaluate_clients=2,       # can be lower than 4 [web:9]
            min_available_clients=4,      # ensure all 4 are connected before starting [web:9]
            evaluate_fn=get_evaluate_fn(args.model),  # optional centralized evaluation [web:24]
            on_fit_config_fn=fit_config,  # per-round client config [web:29]
            proximal_mu=0.01,             # server-side FedProx mu [web:9]
            initial_parameters=initial_parameters,  # seed global weights [web:24][web:29]
        )  # [web:9]
        chosen = "FedProx"  # [web:9]
    else:
        # Alternative robust optimizer: FedAdagrad (adaptive server optimizer) with full client participation. [web:25][web:33]
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=1.0,             # select all clients each round [web:25][web:28]
            fraction_evaluate=1.0,        # evaluate on all clients [web:25]
            min_fit_clients=4,            # expect all 4 clients [web:25]
            min_evaluate_clients=2,       # allow subset for evaluation [web:25]
            min_available_clients=4,      # ensure enough clients are connected [web:25]
            evaluate_fn=get_evaluate_fn(args.model),  # optional centralized evaluation [web:24]
            on_fit_config_fn=fit_config,  # dynamic client config [web:29]
            initial_parameters=initial_parameters,    # seed global weights [web:24][web:29]
            eta=0.1,      # server learning rate [web:25]
            eta_l=0.1,    # client learning rate-like factor in FedOpt family [web:25]
            tau=1e-9,     # numerical stability [web:25]
        )  # [web:25]
        chosen = "FedAdagrad"  # [web:25]

    print(f"Starting Flower server with strategy: {chosen}", flush=True)  # [web:9][web:25]

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )  # [web:24]


if __name__ == "__main__":
    main()  # [web:24]


# [1](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedProx.html)
# [2](https://flower.ai/docs/framework/_modules/flwr/server/strategy/fedprox.html)
# [3](https://apxml.com/courses/federated-learning/chapter-6-federated-learning-system-design/practice-fl-simulation-framework)
# [4](https://trepo.tuni.fi/bitstream/handle/10024/157908/DasariSaiPoojith.pdf?sequence=2)
# [5](https://www.youtube.com/watch?v=yOV4aGBfOTk)
# [6](https://flower.ai/docs/framework/ref-api/flwr.server.start_server.html)
# [7](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAdagrad.html)
# [8](https://akhilmathurs.github.io/papers/beutel_flower2020.pdf)
# [9](https://www.drivendata.org/competitions/140/uk-federated-learning-2-financial-crime-federated/page/638/)
# [10](https://research-information.bris.ac.uk/files/359950483/Hyperparameter_Optimisation_in_Federated_Learning.pdf)
# [11](https://arxiv.org/html/2407.12980v2)
# [12](https://flower.ai/docs/framework/main/fr/how-to-implement-strategies.html)
# [13](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAdagrad.html)
# [14](https://sands.kaust.edu.sa/papers/colext.24.pdf)
# [15](https://github.com/adap/flower/issues/487)
# [16](https://github.com/VectorInstitute/FL4Health)
# [17](https://huggingface.co/blog/fl-with-flower)
# [18](https://www.youtube.com/watch?v=KsMP9dgcLw4)
# [19](https://www.distributedgenomics.ca/posts/federated-learning-candig/)
# [20](https://www.kaggle.com/code/snehilsanyal/federated-learning-tutorial-part-1-with-flower)