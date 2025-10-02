import numpy as np
import tensorflow as tf
from tensorflow import keras
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedProx, FedAvg
from flwr.common import Context, ArrayRecord, ConfigRecord
from typing import List, Tuple, Dict, Optional
import os

def build_global_model(input_shape=(224, 224, 3), num_classes=2):
    """Build the initial global model"""
    model = keras.Sequential([
        # Convolutional Block 1
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Convolutional Block 2
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Convolutional Block 3
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Dense Layers
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Aggregate metrics from clients using weighted average
    Used for aggregating evaluation metrics
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    results = {"accuracy": sum(accuracies) / sum(examples)}
    
    # Also aggregate AUC if available
    if "auc" in metrics[0][1]:
        aucs = [num_examples * m["auc"] for num_examples, m in metrics]
        results["auc"] = sum(aucs) / sum(examples)
    
    return results


def server_fn(context: Context):
    """
    Define the server-side execution logic
    """
    # Read configuration from context
    num_rounds = context.run_config.get("num-server-rounds", 10)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    min_fit_clients = context.run_config.get("min-fit-clients", 4)
    min_evaluate_clients = context.run_config.get("min-evaluate-clients", 4)
    min_available_clients = context.run_config.get("min-available-clients", 4)
    local_epochs = context.run_config.get("local-epochs", 5)
    batch_size = context.run_config.get("batch-size", 32)
    learning_rate = context.run_config.get("learning-rate", 0.001)
    use_fedprox = context.run_config.get("use-fedprox", True)
    proximal_mu = context.run_config.get("proximal-mu", 0.1)
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING SERVER - BREAST CANCER DETECTION")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Strategy: {'FedProx' if use_fedprox else 'FedAvg'}")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Local epochs: {local_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Fraction fit: {fraction_fit}")
    print(f"  - Min fit clients: {min_fit_clients}")
    if use_fedprox:
        print(f"  - Proximal mu: {proximal_mu}")
    print("="*70 + "\n")
    
    # Initialize global model
    global_model = build_global_model()
    initial_parameters = [np.array(w) for w in global_model.get_weights()]
    
    # Create configuration to send to clients
    config = {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": learning_rate
    }
    
    # Choose strategy based on configuration
    if use_fedprox:
        # FedProx: Better for non-IID data
        strategy = FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=proximal_mu,  # Proximal term coefficient
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda server_round: config,
            on_evaluate_config_fn=lambda server_round: {}
        )
        print("✓ Using FedProx strategy (optimized for non-IID data)")
    else:
        # Standard FedAvg
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda server_round: config,
            on_evaluate_config_fn=lambda server_round: {}
        )
        print("✓ Using FedAvg strategy")
    
    # Configure server
    server_config = ServerConfig(num_rounds=num_rounds)
    
    return strategy, server_config


def save_final_model(parameters, save_path="federated_breast_cancer_model.h5"):
    """Save the final global model"""
    model = build_global_model()
    model.set_weights(parameters)
    model.save(save_path)
    print(f"\n✓ Final model saved to: {save_path}")


# Create ServerApp
app = ServerApp(server_fn=server_fn)
