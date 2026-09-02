"""
Visualization Utilities
Plot training history, confusion matrices, distributions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(history, title='Training History'):
    """
    Plot training and validation metrics over epochs
    
    Args:
        history (dict): Dictionary with keys:
            - 'train_losses': list of training losses
            - 'val_losses': list of validation losses
            - 'val_accuracies': list of validation accuracies
        title (str): Plot title
    """
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss plot
    axes[0].plot(epochs, history['train_losses'], label='Train Loss', marker='o')
    axes[0].plot(epochs, history['val_losses'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['val_accuracies'], label='Val Accuracy', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix, class_names=None, title='Confusion Matrix'):
    """
    Plot confusion matrix as heatmap
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix
        class_names (dict): Class name mapping
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Prepare labels
    if class_names:
        labels = [class_names.get(i, f'Class {i}') for i in range(len(conf_matrix))]
    else:
        labels = [f'Class {i}' for i in range(len(conf_matrix))]
    
    # Plot heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(confidence_scores, title='Confidence Distribution'):
    """
    Plot distribution of confidence scores
    
    Args:
        confidence_scores (np.ndarray or torch.Tensor): Confidence scores
        title (str): Plot title
    """
    if hasattr(confidence_scores, 'cpu'):
        confidence_scores = confidence_scores.cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    
    plt.hist(confidence_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(confidence_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidence_scores.mean():.3f}')
    plt.axvline(np.median(confidence_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(confidence_scores):.3f}')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_modality_weights(weights_history, title='Modality Weights Over Time'):
    """
    Plot evolution of fusion weights (speech vs face)
    
    Args:
        weights_history (dict): Dictionary with:
            - 'speech_weights': list of speech weights
            - 'face_weights': list of face weights
        title (str): Plot title
    """
    batches = range(len(weights_history['speech_weights']))
    
    plt.figure(figsize=(12, 5))
    
    plt.plot(batches, weights_history['speech_weights'], label='Speech Weight', alpha=0.7)
    plt.plot(batches, weights_history['face_weights'], label='Face Weight', alpha=0.7)
    
    plt.xlabel('Batch')
    plt.ylabel('Weight')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Equal weight')
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels, class_names=None, title='Class Distribution'):
    """
    Plot distribution of emotion classes
    
    Args:
        labels (np.ndarray or torch.Tensor): Class labels
        class_names (dict): Class name mapping
        title (str): Plot title
    """
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    unique, counts = np.unique(labels, return_counts=True)
    
    # Prepare class names
    if class_names:
        class_labels = [class_names.get(int(i), f'Class {i}') for i in unique]
    else:
        class_labels = [f'Class {i}' for i in unique]
    
    plt.figure(figsize=(10, 5))
    
    bars = plt.bar(class_labels, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom')
    
    plt.xlabel('Emotion Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
