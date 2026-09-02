"""
Evaluation Metrics
Compute accuracy, precision, recall, F1, confusion matrix
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def compute_metrics(predictions, true_labels, class_names=None, average='weighted'):
    """
    Compute classification metrics
    
    Args:
        predictions (np.ndarray or torch.Tensor): Predicted class indices
        true_labels (np.ndarray or torch.Tensor): True class labels
        class_names (dict): Optional class name mapping
        average (str): 'weighted', 'macro', 'micro'
    
    Returns:
        metrics (dict): Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average=average, zero_division=0)
    recall = recall_score(true_labels, predictions, average=average, zero_division=0)
    f1 = f1_score(true_labels, predictions, average=average, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print metrics in readable format
    
    Args:
        metrics (dict): Metrics dictionary from compute_metrics
        class_names (dict): Optional class name mapping
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Per-class metrics if we have confusion matrix
    if class_names and len(metrics['confusion_matrix']) == len(class_names):
        print(f"\nPer-class metrics:")
        conf_matrix = metrics['confusion_matrix']
        
        for i, class_name in class_names.items():
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  Precision: {class_precision:.4f}")
            print(f"  Recall:    {class_recall:.4f}")
            print(f"  F1-Score:  {class_f1:.4f}")
    
    print("="*60)


def get_per_class_metrics(true_labels, predictions, class_names=None):
    """
    Get per-class precision, recall, F1
    
    Args:
        true_labels (np.ndarray): True labels
        predictions (np.ndarray): Predicted labels
        class_names (dict): Optional class names
    
    Returns:
        per_class_metrics (dict): Metrics for each class
    """
    conf_matrix = confusion_matrix(true_labels, predictions)
    num_classes = conf_matrix.shape[0]
    
    per_class_metrics = {}
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = class_names.get(i, f'Class {i}') if class_names else f'Class {i}'
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': conf_matrix[i].sum()
        }
    
    return per_class_metrics
