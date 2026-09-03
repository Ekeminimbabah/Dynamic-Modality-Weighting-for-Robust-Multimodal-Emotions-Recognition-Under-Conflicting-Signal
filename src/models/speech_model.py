"""
Speech Emotion Recognition Model
Includes model definition, training loop, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import sys
from pathlib import Path


# MODEL DEFINITION

class SpeechEmotionModel(nn.Module):
    """
    CNN for speech emotion recognition with embedding extraction support
    
    Input: MFCC features (batch_size, 1, 13, 384)
    Output:
      - Logits (batch_size, 4) if return_embeddings=False
      - Embeddings (batch_size, 128) if return_embeddings=True
    """
    
    def __init__(self, num_emotions=4, input_channels=1):
        super(SpeechEmotionModel, self).__init__()
        
        # ===== CONVOLUTIONAL LAYERS =====
        # Extract temporal and spectral patterns from MFCC features.
        # Similar architecture to face model but processes (MFCC+Delta, time) instead of (height, width).
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            # Capture local temporal-spectral patterns in lower frequencies
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            # Extract medium-level prosodic features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            # Capture high-level emotional cues from speech
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # ===== FEATURE EXTRACTION =====
        # After 3 max pools: 13x384 -> 1x48
        # Compress spectral features into 128-dimensional embedding matching face modality.
        # This ensures compatible representations for multimodal fusion concatenation.
        self.feature_layers = nn.Sequential(
            # Reduce spatial/temporal dimensions to semantic embedding
            nn.Linear(128 * 1 * 48, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Project to 128D embedding space aligned with face embeddings
            nn.Linear(256, 128),  # Embedding dimension: 128
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # ===== CLASSIFICATION LAYER =====
        # Separate classifier allows dual output: embeddings for fusion or logits for evaluation.
        self.classifier = nn.Linear(128, num_emotions)
    
    def forward(self, x, return_embeddings=False):
        """
        Forward pass through the model
        
        Args:
            x (torch.Tensor): Input (batch_size, 1, 128, 128)
            return_embeddings (bool): If True, return embeddings before classification
        
        Returns:
            torch.Tensor: Logits (batch_size, 4) or Embeddings (batch_size, 128)
        """
        # Extract acoustic features via convolutional layers
        x = self.conv_layers(x)
        
        # Reshape for fully connected layers: [batch, channels, freq, time] -> [batch, features]
        x = x.view(x.size(0), -1)
        
        # Generate 128-dimensional speech representation compatible with face embeddings
        embeddings = self.feature_layers(x)
        
        # Conditional output: embeddings for multimodal fusion, logits for standalone evaluation
        if return_embeddings:
            return embeddings
        
        # Predict emotion class from acoustic representation
        logits = self.classifier(embeddings)
        return logits


# TRAINING LOOP

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: CPU or GPU
    
    Returns:
        Average loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Move data to device (CPU or GPU)
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, device='cpu'):
    """
    Complete training loop for the model
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: CPU or GPU
    
    Returns:
        Dictionary containing training and validation metrics
    """
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to device
    model = model.to(device)
    
    # Track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    print(f"Device: {device}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}\n")
    
    # Return training history
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


# EVALUATION

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: Neural network model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: CPU or GPU
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def detailed_evaluation(model, test_loader, device, emotion_labels=None):
    """
    Detailed evaluation with multiple metrics and confusion matrix
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: CPU or GPU
        emotion_labels: List of emotion label names (e.g., ['happy', 'sad', 'angry', 'neutral'])
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    if emotion_labels:
        print("\nPer-class metrics:")
        for i, label in enumerate(emotion_labels):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            print(f"\n{label}:")
            print(f"  Precision: {class_precision:.4f}")
            print(f"  Recall:    {class_recall:.4f}")
            print(f"  F1-Score:  {class_f1:.4f}")
    
    return results


def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary from train_model() containing loss and accuracy history
    """
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(epochs, history['train_losses'], label='Train Loss')
    axes[0].plot(epochs, history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, history['val_accuracies'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


# EXAMPLE USAGE

if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Example: Create model
    model = SpeechEmotionModel(num_emotions=4, input_channels=1)
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    print("Model Architecture:")
    print(model)
    
    # Note: To use the model with actual data:
    # 1. Prepare your audio features (mel-spectrograms or MFCC)
    # 2. Create DataLoaders for training, validation, and test sets
    # 3. Call train_model() with your data
    # 4. Call detailed_evaluation() on test set
    # 5. Call plot_training_history() to visualize results
