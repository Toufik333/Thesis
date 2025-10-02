Here’s a complete, project-level documentation (README) tailored to the provided server, client, and orchestration scripts.

# Federated Breast Cancer Detection with Flower (PyTorch)

A minimal federated learning project using Flower and PyTorch to train a binary classifier for breast cancer detection across four simulated non-IID clients. The project uses a simple tabular-like synthetic dataset, an MLP model, a FedProx or FedAdagrad server strategy, and a subprocess-based launcher.

## Project structure

- client.py
  - Flower NumPyClient implementation for local training/evaluation.
  - Simulates non-IID data partitions across 4 clients via label-skewed sampling.
  - Implements a simple MLP (and optional CNN) for binary classification.
  - Supports an optional FedProx proximal term during local training.

- server.py
  - Flower server coordinating training with a non-FedAvg strategy.
  - Default strategy: FedProx; alternative: FedAdagrad via CLI flag.
  - Sends dynamic per-round config (local epochs, batch size, FedProx mu).
  - Optional placeholder centralized evaluation hook.

- main.py
  - Orchestrates starting server first, then four clients with unique IDs 0..3.
  - Manages processes using subprocess and ensures cleanup.

## Features

- Non-IID data simulation
  - Each client receives a different class proportion (e.g., 85/15, 15/85, 70/30, 30/70) to mimic heterogeneity.

- Model
  - Default: MLP for 30 input features, suitable for tabular data.
  - Optional CNN path to demonstrate flexibility (maps 30 features into 1×6×5).

- Training
  - BCEWithLogitsLoss with accuracy metric.
  - Optional FedProx proximal regularization in the client’s fit loop.
  - Configurable local epochs per round provided by the server.

- Server strategies
  - FedProx (default): fraction_fit=1.0 ensures all 4 clients participate each round.
  - FedAdagrad (alternative): adaptive server optimizer robust to non-IID.

## Requirements

- Python 3.9+
- PyTorch
- Flower
- NumPy

Example installation:
- pip install torch flwr numpy

Note: GPU is optional; CPU-only is supported.

## How it works

1. The server starts and waits for clients to connect.
2. Each round:
   - The server broadcasts global parameters and per-round config (e.g., local_epochs, proximal_mu).
   - All four clients perform local training with their non-IID data and return updated parameters and metrics.
   - The server aggregates updates using the chosen strategy (FedProx or FedAdagrad).
3. After 10 rounds, training stops and processes exit.

## Usage

Option A: Orchestrated run (recommended for local testing)
- python main.py

Notes:
- main.py starts server.py, waits briefly, then launches client.py with client_id 0..3.
- Adjust the wait duration in main.py if the environment starts slowly.

Option B: Manual run (separate terminals)
1) Server:
- python server.py --server_address [::]:8080 --num_rounds 10 --strategy fedprox --model mlp

2) Clients (four terminals or background processes):
- python client.py --client_id 0 --server_address [::]:8080
- python client.py --client_id 1 --server_address [::]:8080
- python client.py --client_id 2 --server_address [::]:8080
- python client.py --client_id 3 --server_address [::]:8080

Tip:
- Use the same address literal on both server and clients to avoid IPv4/IPv6 mismatches. For local-only runs, consistent choices include [::]:8080 or 127.0.0.1:8080.

## Configuration

Server (server.py):
- Strategy selection:
  - --strategy fedprox (default)
  - --strategy fedadagrad
- Rounds: --num_rounds 10
- Address: --server_address [::]:8080
- Model type for eval/init: --model mlp or --model cnn

Dynamic fit config (per round):
- local_epochs: increases with server_round (1 → 2 → 3)
- batch_size: 32
- proximal_mu: 0.01 (FedProx penalty coefficient; used by client fit if > 0)

Client (client.py):
- --client_id 0..3
- --server_address host:port
- --model mlp or --model cnn

Model details:
- MLPBinary(in_features=30, hidden=64)
- SimpleCNN(in_channels=1, 6×5) if cnn is explicitly requested

## Non-IID data simulation

- Synthetic tabular data with 30 features is generated once per client run.
- Label skew by client:
  - Client 0: 85% benign (0), 15% malignant (1)
  - Client 1: 15% benign, 85% malignant
  - Client 2: 70% benign, 30% malignant
  - Client 3: 30% benign, 70% malignant
- Each client splits its local subset into train/val/test.

## FedProx in client training

- The client fit loop includes an optional proximal term:
  - loss_total = BCEWithLogitsLoss + (mu/2) * Σ ||w_local − w_global||^2
- mu is read from server config (proximal_mu).
- Set proximal_mu to 0.0 to disable the proximal term.

## Logging and metrics

Client fit returns:
- train_loss, train_acc from the last local epoch of the round
- val_loss, val_acc (client-side validation)
- client_id

Server evaluate_fn:
- Placeholder returning a constant loss; replace with centralized validation if needed.

## Tips and troubleshooting

- Address consistency:
  - Make sure server.py and client.py use the same exact address literal.
  - If using main.py, consider making both server and clients use [::]:8080 or 127.0.0.1:8080 consistently.

- Port availability:
  - Ensure port 8080 is free.
  - Firewalls may need to allow local traffic for the chosen port.

- Versions:
  - Keep Flower and PyTorch versions compatible.
  - If using a very new Flower release that marks start_server as deprecated, pin to a compatible version or plan to migrate to newer orchestration patterns later.

- Runtime timeouts:
  - If clients start before the server is listening, increase the sleep in main.py.
  - If some clients exit early, verify that all four are launched and remain connected until training completes.

- CPU/GPU:
  - The client auto-selects CUDA if available; otherwise falls back to CPU.

## Extending the project

- Replace synthetic data with real data loaders:
  - Implement a local CSV or parquet reader with deterministic partitioning.
  - Keep label skew logic to maintain non-IID scenarios.

- Add centralized evaluation:
  - In server.py, replace the placeholder evaluate_fn with a loader over a held-out validation set.

- Experiment with strategies:
  - Try FedYogi or other FedOpt strategies by adjusting server.py.
  - Tune proximal_mu, learning rates, and local_epochs schedules.

- Model improvements:
  - Add feature normalization, class weighting, or focal loss for imbalance.
  - Replace MLP with more advanced architectures suitable for tabular data.

## Security and privacy notes

- This example simulates data locally for demonstration only.
- Real deployments should consider secure aggregation, TLS, authentication, DP, and policy controls as appropriate to the domain.

## License

- This example is intended for educational and research purposes.
- Ensure compliance with all dependencies’ licenses when distributing.

[1](https://flower.ai/docs/)
[2](https://flower.ai/docs/framework/index.html)
[3](https://github.com/adap/flower)
[4](https://huggingface.co/blog/fl-with-flower)
[5](https://github.com/n3pt7un/Federated-Learning-LR_RF)
[6](https://www.youtube.com/watch?v=XK_dRVcSZqg)
[7](https://colab.research.google.com/github/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb)