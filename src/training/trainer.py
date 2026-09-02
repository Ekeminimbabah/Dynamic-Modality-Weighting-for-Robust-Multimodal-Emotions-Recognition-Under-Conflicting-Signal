"""
Training Pipeline for Multimodal Emotion Recognition
Handles training loops, validation, and testing for individual and fused models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json


# =====================================================================
# TRAINING UTILITIES
# =====================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainingState:
    """Tracks training progress and metrics"""
    
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
    
    def save_checkpoint(self, model, optimizer, save_path, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
        }
        
        torch.save(checkpoint, save_path)
        if is_best:
            best_path = save_path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_f1_scores = checkpoint['val_f1_scores']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, optimizer


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_epoch_single_modality(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    modality_name: str = "Model"
) -> float:
    """
    Train a single modality model for one epoch (Speech or Face)
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function (typically CrossEntropyLoss)
        optimizer: Optimization algorithm
        device: CPU or GPU
        modality_name: Name for logging (e.g., "Speech" or "Face")
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Move data to device
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress logging (every 10 batches)
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{modality_name}] Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_epoch_multimodal(
    speech_model: nn.Module,
    face_model: nn.Module,
    fusion_model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    freeze_individual_models: bool = True
) -> float:
    """
    Train multimodal fusion model for one epoch
    
    Args:
        speech_model: Trained speech emotion model
        face_model: Trained face emotion model
        fusion_model: Multimodal fusion model
        train_loader: DataLoader with (speech_features, face_features, labels)
        criterion: Loss function
        optimizer: Optimization algorithm
        device: CPU or GPU
        freeze_individual_models: If True, freeze speech/face models and only train fusion
    
    Returns:
        Average loss for the epoch
    """
    if freeze_individual_models:
        speech_model.eval()
        face_model.eval()
    else:
        speech_model.train()
        face_model.train()
    
    fusion_model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Handle different batch formats
        if len(batch) == 3:
            speech_features, face_features, labels = batch
        elif len(batch) == 2 and isinstance(batch[0], dict):
            # Dictionary format
            speech_features = batch[0]['speech']
            face_features = batch[0]['face']
            labels = batch[1]
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} elements")
        
        speech_features = speech_features.to(device)
        face_features = face_features.to(device)
        labels = labels.to(device)
        
        # Get logits from individual models (no gradients if frozen)
        with torch.no_grad() if freeze_individual_models else torch.enable_grad():
            speech_logits = speech_model(speech_features)
            face_logits = face_model(face_features)
        
        # Forward through fusion model
        fused_logits = fusion_model(speech_logits, face_logits)
        loss = criterion(fused_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  [Multimodal] Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


# =====================================================================
# EVALUATION FUNCTIONS
# =====================================================================

def evaluate_single_modality(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    modality_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a single modality model on validation/test set
    
    Args:
        model: Neural network model
        data_loader: DataLoader for validation/test data
        device: CPU or GPU
        criterion: Loss function (optional, for computing loss)
        modality_name: Name for logging
    
    Returns:
        Dictionary with keys: 'loss', 'accuracy', 'precision', 'recall', 'f1_score'
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(features)
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    print(f"[{modality_name}] Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
          f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
    
    return metrics


def evaluate_multimodal(
    speech_model: nn.Module,
    face_model: nn.Module,
    fusion_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_weights: bool = False
) -> Dict:
    """
    Evaluate multimodal fusion model on validation/test set
    
    Args:
        speech_model: Trained speech emotion model
        face_model: Trained face emotion model
        fusion_model: Multimodal fusion model
        data_loader: DataLoader with (speech_features, face_features, labels)
        device: CPU or GPU
        criterion: Loss function (optional)
        return_weights: If True, return dynamic weights for each sample
    
    Returns:
        Dictionary with metrics and optionally weights
    """
    speech_model.eval()
    face_model.eval()
    fusion_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    all_weights_speech = []
    all_weights_face = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different batch formats
            if len(batch) == 3:
                speech_features, face_features, labels = batch
            elif len(batch) == 2 and isinstance(batch[0], dict):
                speech_features = batch[0]['speech']
                face_features = batch[0]['face']
                labels = batch[1]
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            
            speech_features = speech_features.to(device)
            face_features = face_features.to(device)
            labels = labels.to(device)
            
            # Get logits from individual models
            speech_logits = speech_model(speech_features)
            face_logits = face_model(face_features)
            
            # Forward through fusion model
            if return_weights:
                fused_logits, weights = fusion_model(
                    speech_logits, face_logits, return_weights=True
                )
                all_weights_speech.append(weights['w_s'].cpu().numpy())
                all_weights_face.append(weights['w_f'].cpu().numpy())
            else:
                fused_logits = fusion_model(speech_logits, face_logits, return_weights=False)
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(fused_logits, labels)
                total_loss += loss.item()
                num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(fused_logits, dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    if return_weights:
        metrics['weights_speech'] = np.concatenate(all_weights_speech)
        metrics['weights_face'] = np.concatenate(all_weights_face)
    
    print(f"[Multimodal] Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
          f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
    
    return metrics


def compute_per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute per-class metrics (precision, recall, F1)
    
    Args:
        predictions: Predicted class indices
        labels: True class labels
        class_names: Optional names for classes
    
    Returns:
        Dictionary with per-class metrics
    """
    num_classes = len(np.unique(labels))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    per_class_metrics = {}
    
    for class_idx in range(num_classes):
        # Binary classification: this class vs. rest
        binary_pred = (predictions == class_idx).astype(int)
        binary_true = (labels == class_idx).astype(int)
        
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        
        per_class_metrics[class_names[class_idx]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(np.sum(labels == class_idx))
        }
    
    return per_class_metrics


# =====================================================================
# TRAINING ORCHESTRATION
# =====================================================================

class SingleModalityTrainer:
    """Trainer for single modality models (Speech or Face)"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        modality_name: str = "Model",
        checkpoint_dir: Optional[str] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.modality_name = modality_name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training state
        self.state = TrainingState()
    
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Stop if validation loss doesn't improve for this many epochs
            verbose: Print detailed progress
        
        Returns:
            Dictionary with training history
        """
        print(f"\n{'='*60}")
        print(f"Training {self.modality_name}")
        print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            self.state.epoch = epoch + 1
            
            # Train
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()
            
            train_loss = train_epoch_single_modality(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.modality_name
            )
            self.state.train_losses.append(train_loss)
            
            # Validate
            val_metrics = evaluate_single_modality(
                self.model, self.val_loader, self.device,
                self.criterion, self.modality_name
            )
            self.state.val_losses.append(val_metrics['loss'])
            self.state.val_accuracies.append(val_metrics['accuracy'])
            self.state.val_f1_scores.append(val_metrics['f1_score'])
            
            epoch_time = time.time() - start_time
            print(f"  Time: {epoch_time:.2f}s")
            
            # Check if best validation loss
            if val_metrics['loss'] < self.state.best_val_loss:
                self.state.best_val_loss = val_metrics['loss']
                self.state.best_val_acc = val_metrics['accuracy']
                self.state.patience_counter = 0
                
                # Save best checkpoint
                if self.checkpoint_dir:
                    checkpoint_path = self.checkpoint_dir / f"{self.modality_name}_checkpoint.pt"
                    self.state.save_checkpoint(
                        self.model, self.optimizer, str(checkpoint_path), is_best=True
                    )
                    print(f"  ✓ Saved best checkpoint")
            else:
                self.state.patience_counter += 1
                if self.state.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Step learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Loss: {self.state.best_val_loss:.4f}")
        print(f"Best Val Acc:  {self.state.best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        return {
            'train_losses': self.state.train_losses,
            'val_losses': self.state.val_losses,
            'val_accuracies': self.state.val_accuracies,
            'val_f1_scores': self.state.val_f1_scores,
            'best_val_loss': self.state.best_val_loss,
            'best_val_acc': self.state.best_val_acc
        }
    
    def load_best_model(self):
        """Load the best saved checkpoint"""
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / f"{self.modality_name}_checkpoint_best.pt"
            if checkpoint_path.exists():
                self.model, self.optimizer = self.state.load_checkpoint(
                    self.model, self.optimizer, str(checkpoint_path)
                )
                print(f"Loaded best {self.modality_name} checkpoint")


class MultimodalTrainer:
    """Trainer for multimodal fusion model"""
    
    def __init__(
        self,
        speech_model: nn.Module,
        face_model: nn.Module,
        fusion_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        freeze_individual_models: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        self.speech_model = speech_model
        self.face_model = face_model
        self.fusion_model = fusion_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.freeze_individual_models = freeze_individual_models
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer (only for fusion model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.fusion_model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.state = TrainingState()
    
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """Train the multimodal fusion model"""
        print(f"\n{'='*60}")
        print(f"Training Multimodal Fusion")
        print(f"{'='*60}")
        print(f"Freeze Individual Models: {self.freeze_individual_models}")
        
        for epoch in range(num_epochs):
            self.state.epoch = epoch + 1
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()
            
            train_loss = train_epoch_multimodal(
                self.speech_model, self.face_model, self.fusion_model,
                self.train_loader, self.criterion, self.optimizer,
                self.device, self.freeze_individual_models
            )
            self.state.train_losses.append(train_loss)
            
            # Validate
            val_metrics = evaluate_multimodal(
                self.speech_model, self.face_model, self.fusion_model,
                self.val_loader, self.device, self.criterion
            )
            self.state.val_losses.append(val_metrics['loss'])
            self.state.val_accuracies.append(val_metrics['accuracy'])
            self.state.val_f1_scores.append(val_metrics['f1_score'])
            
            epoch_time = time.time() - start_time
            print(f"  Time: {epoch_time:.2f}s")
            
            # Check if best
            if val_metrics['loss'] < self.state.best_val_loss:
                self.state.best_val_loss = val_metrics['loss']
                self.state.best_val_acc = val_metrics['accuracy']
                self.state.patience_counter = 0
                
                if self.checkpoint_dir:
                    checkpoint_path = self.checkpoint_dir / "multimodal_checkpoint.pt"
                    self.state.save_checkpoint(
                        self.fusion_model, self.optimizer, str(checkpoint_path), is_best=True
                    )
                    print(f"  ✓ Saved best checkpoint")
            else:
                self.state.patience_counter += 1
                if self.state.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            self.scheduler.step(val_metrics['loss'])
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Loss: {self.state.best_val_loss:.4f}")
        print(f"Best Val Acc:  {self.state.best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        return {
            'train_losses': self.state.train_losses,
            'val_losses': self.state.val_losses,
            'val_accuracies': self.state.val_accuracies,
            'val_f1_scores': self.state.val_f1_scores,
            'best_val_loss': self.state.best_val_loss,
            'best_val_acc': self.state.best_val_acc
        }
    
    def load_best_model(self):
        """Load the best saved checkpoint"""
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / "multimodal_checkpoint_best.pt"
            if checkpoint_path.exists():
                self.fusion_model, self.optimizer = self.state.load_checkpoint(
                    self.fusion_model, self.optimizer, str(checkpoint_path)
                )
                print(f"Loaded best multimodal checkpoint")


# =====================================================================
# COMPREHENSIVE EVALUATION
# =====================================================================

def run_full_evaluation(
    speech_model: nn.Module,
    face_model: nn.Module,
    fusion_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict:
    """
    Run comprehensive evaluation on test set
    Evaluates speech model, face model, and multimodal fusion
    
    Args:
        speech_model: Trained speech emotion model
        face_model: Trained face emotion model
        fusion_model: Multimodal fusion model
        test_loader: DataLoader with test data
        device: CPU or GPU
        class_names: Optional emotion class names
        save_path: Optional path to save results as JSON
    
    Returns:
        Dictionary with all evaluation results
    """
    if class_names is None:
        class_names = ["Angry", "Happy", "Neutral", "Sad"]
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST SET EVALUATION")
    print(f"{'='*60}\n")
    
    # Evaluate individual models
    print("Evaluating Speech Model...")
    speech_metrics = evaluate_single_modality(
        speech_model, test_loader, device, modality_name="Speech"
    )
    speech_per_class = compute_per_class_metrics(
        speech_metrics['predictions'], speech_metrics['labels'], class_names
    )
    
    print("\nEvaluating Face Model...")
    face_metrics = evaluate_single_modality(
        face_model, test_loader, device, modality_name="Face"
    )
    face_per_class = compute_per_class_metrics(
        face_metrics['predictions'], face_metrics['labels'], class_names
    )
    
    print("\nEvaluating Multimodal Fusion...")
    multimodal_metrics = evaluate_multimodal(
        speech_model, face_model, fusion_model,
        test_loader, device, return_weights=True
    )
    multimodal_per_class = compute_per_class_metrics(
        multimodal_metrics['predictions'], multimodal_metrics['labels'], class_names
    )
    
    # Compile results
    results = {
        'speech_model': {
            'overall_metrics': {
                'accuracy': float(speech_metrics['accuracy']),
                'precision': float(speech_metrics['precision']),
                'recall': float(speech_metrics['recall']),
                'f1_score': float(speech_metrics['f1_score']),
                'loss': float(speech_metrics['loss'])
            },
            'per_class_metrics': speech_per_class
        },
        'face_model': {
            'overall_metrics': {
                'accuracy': float(face_metrics['accuracy']),
                'precision': float(face_metrics['precision']),
                'recall': float(face_metrics['recall']),
                'f1_score': float(face_metrics['f1_score']),
                'loss': float(face_metrics['loss'])
            },
            'per_class_metrics': face_per_class
        },
        'multimodal_fusion': {
            'overall_metrics': {
                'accuracy': float(multimodal_metrics['accuracy']),
                'precision': float(multimodal_metrics['precision']),
                'recall': float(multimodal_metrics['recall']),
                'f1_score': float(multimodal_metrics['f1_score']),
                'loss': float(multimodal_metrics['loss'])
            },
            'per_class_metrics': multimodal_per_class,
            'avg_weight_speech': float(np.mean(multimodal_metrics['weights_speech'])),
            'avg_weight_face': float(np.mean(multimodal_metrics['weights_face']))
        }
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\nSpeech Model Performance:")
    print(f"  Accuracy: {results['speech_model']['overall_metrics']['accuracy']:.4f}")
    print(f"  F1-Score: {results['speech_model']['overall_metrics']['f1_score']:.4f}")
    
    print("\nFace Model Performance:")
    print(f"  Accuracy: {results['face_model']['overall_metrics']['accuracy']:.4f}")
    print(f"  F1-Score: {results['face_model']['overall_metrics']['f1_score']:.4f}")
    
    print("\nMultimodal Fusion Performance:")
    print(f"  Accuracy: {results['multimodal_fusion']['overall_metrics']['accuracy']:.4f}")
    print(f"  F1-Score: {results['multimodal_fusion']['overall_metrics']['f1_score']:.4f}")
    print(f"  Avg Speech Weight: {results['multimodal_fusion']['avg_weight_speech']:.4f}")
    print(f"  Avg Face Weight:   {results['multimodal_fusion']['avg_weight_face']:.4f}")
    
    print(f"\n{'='*60}\n")
    
    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {save_path}")
    
    return results
