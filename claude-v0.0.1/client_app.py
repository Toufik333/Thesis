import numpy as np
import tensorflow as tf
from tensorflow import keras
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ArrayRecord, ConfigRecord, MetricRecord
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Tuple, Dict
import os

class BreastCancerClient(NumPyClient):
    """
    Flower NumPyClient for breast cancer detection
    Handles non-IID datasets with local training
    """
    
    def __init__(self, model, X_train, y_train, X_val, y_val, client_id):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.client_id = client_id
        
    def get_parameters(self, config):
        """Return current model parameters"""
        return [np.array(w) for w in self.model.get_weights()]
    
    def fit(self, parameters, config):
        """Train model with data of this client"""
        # Update local model with global parameters
        self.model.set_weights(parameters)
        
        # Get training configuration
        epochs = config.get("local_epochs", 5)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("lr", 0.001)
        
        # Set learning rate
        self.model.optimizer.learning_rate = learning_rate
        
        print(f"\n[Client {self.client_id}] Training for {epochs} epochs...")
        
        # Train the model
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            verbose=1
        )
        
        # Get training metrics
        train_loss = float(history.history['loss'][-1])
        train_accuracy = float(history.history['accuracy'][-1])
        
        print(f"[Client {self.client_id}] Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Return updated model parameters and metrics
        return (
            self.get_parameters(config={}),
            len(self.X_train),  # Number of examples used for training
            {"train_loss": train_loss, "train_accuracy": train_accuracy}
        )
    
    def evaluate(self, parameters, config):
        """Evaluate model on client's local data"""
        # Update model with parameters from server
        self.model.set_weights(parameters)
        
        # Evaluate on validation set
        loss, accuracy, auc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Return loss, number of examples, and metrics
        return (
            float(loss),
            len(self.X_val),
            {"accuracy": float(accuracy), "auc": float(auc)}
        )


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build CNN model for breast cancer detection
    Optimized architecture for medical imaging
    """
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


def load_client_data(dataset_path: str, client_id: int) -> Tuple:
    """
    Load and preprocess client's non-IID dataset
    
    Args:
        dataset_path: Path to the dataset file
        client_id: ID of the client
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    print(f"\n[Client {client_id}] Loading dataset from: {dataset_path}")
    
    try:
        # Option 1: Load from NumPy file
        if dataset_path.endswith('.npy') or dataset_path.endswith('.npz'):
            data = np.load(dataset_path, allow_pickle=True)
            
            # Handle different formats
            if isinstance(data, np.ndarray):
                # Assume first column is data, second is labels
                X = data[:, 0]
                y = data[:, 1]
            else:
                # Dictionary format
                X = data['images']
                y = data['labels']
        
        # Option 2: Load from directory structure
        elif os.path.isdir(dataset_path):
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            train_generator = datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            val_generator = datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            
            # Convert generators to arrays
            X_train, y_train = [], []
            for batch_x, batch_y in train_generator:
                X_train.append(batch_x)
                y_train.append(batch_y)
                if len(X_train) * 32 >= train_generator.samples:
                    break
            
            X_val, y_val = [], []
            for batch_x, batch_y in val_generator:
                X_val.append(batch_x)
                y_val.append(batch_y)
                if len(X_val) * 32 >= val_generator.samples:
                    break
            
            X_train = np.vstack(X_train)
            y_train = np.vstack(y_train)
            X_val = np.vstack(X_val)
            y_val = np.vstack(y_val)
            
            return X_train, y_train, X_val, y_val
        
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        # Ensure X is properly shaped
        if len(X.shape) == 1:
            X = np.array([img for img in X])
        
        # Normalize images to [0, 1]
        if X.max() > 1.0:
            X = X.astype('float32') / 255.0
        else:
            X = X.astype('float32')
        
        # Ensure labels are in categorical format
        if len(y.shape) == 1:
            y = keras.utils.to_categorical(y, num_classes=2)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y.argmax(axis=1) if len(y.shape) > 1 else y
        )
        
        # Shuffle training data (important for non-IID)
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        
        print(f"[Client {client_id}] Dataset loaded successfully!")
        print(f"[Client {client_id}] Training samples: {len(X_train)}")
        print(f"[Client {client_id}] Validation samples: {len(X_val)}")
        print(f"[Client {client_id}] Image shape: {X_train[0].shape}")
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        print(f"[Client {client_id}] Error loading dataset: {e}")
        raise


def client_fn(context: Context):
    """
    Create and return a Flower Client instance
    This function is called by the Flower framework for each client
    """
    # Get client ID and dataset path from context
    partition_id = context.node_config["partition-id"]
    dataset_path = context.node_config["dataset-path"]
    
    # Load client's data
    X_train, y_train, X_val, y_val = load_client_data(dataset_path, partition_id)
    
    # Build model
    model = build_model(
        input_shape=X_train[0].shape,
        num_classes=y_train.shape[1]
    )
    
    # Create and return client
    return BreastCancerClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        client_id=partition_id
    ).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
