"""
Confidence Score Utilities
Compute and analyze model confidence for emotion predictions
Implements entropy-based confidence (primary) and max-probability (ablation)
"""

import torch
import numpy as np


def get_entropy_based_confidence(logits, num_classes=None):
    """
    Compute entropy-based confidence scores [PRIMARY METHOD]
    
    Entropy measures prediction uncertainty: low entropy = high confidence.
    Used in dynamic weighting to adaptively adjust modality contributions:
    modalities with low entropy (confident predictions) receive higher weights during fusion.
    
    Formula:
        H(p) = -Σ(i=1 to C) p_i * log(p_i)  [Information entropy]
        c = 1 - H(p) / log(C)  [Normalized confidence score]
    
    Interpretation:
        - c = 1.0: Model completely confident (one-hot prediction)
        - c = 0.0: Model completely uncertain (uniform distribution)
        - Range: [0, 1]
    
    Args:
        logits (torch.Tensor or np.ndarray): Model output logits, shape (batch_size, num_classes)
        num_classes (int): Number of classes. If None, inferred from logits
    
    Returns:
        confidence (np.ndarray): Confidence scores, shape (batch_size,), range [0, 1]
    """
    # Convert to torch if needed
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    
    if num_classes is None:
        num_classes = logits.shape[1]
    
    # Convert logits to probability distribution via softmax
    softmax = torch.softmax(logits, dim=1)
    
    # Calculate information entropy: measures average surprisal of predicted distribution
    # Uniform distribution → max entropy (model uncertain)
    # One-hot distribution → min entropy (model certain)
    entropy = -torch.sum(softmax * torch.log(softmax + 1e-8), dim=1)
    
    # Reference entropy of uniform distribution: maximum possible entropy
    max_entropy = np.log(num_classes)
    
    # Normalize to [0,1]: c=1 for certain predictions, c=0 for completely uncertain
    # Critical for multimodal fusion: confidence directly scales modality contribution
    confidence = 1.0 - (entropy.cpu().numpy() / max_entropy)
    
    return confidence


def get_max_probability_confidence(logits):
    """
    Compute max probability confidence scores [ABLATION BASELINE]
    
    Formula:
        c = max(softmax(logits))
    
    Note: Not recommended for main model. Use entropy-based method instead.
    Kept for ablation study comparison.
    
    Args:
        logits (torch.Tensor or np.ndarray): Model output logits, shape (batch_size, num_classes)
    
    Returns:
        confidence (np.ndarray): Confidence scores, shape (batch_size,), range [0, 1]
    """
    # Convert to torch if needed
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    
    # Compute softmax probabilities
    softmax = torch.softmax(logits, dim=1)
    
    # Get maximum probability
    confidence, _ = torch.max(softmax, dim=1)
    
    return confidence.cpu().numpy()


def get_confidence_scores(logits, method='entropy', num_classes=None):
    """
    Get confidence scores with flexible method selection
    
    Args:
        logits (torch.Tensor or np.ndarray): Model output logits, shape (batch_size, num_classes)
        method (str): Confidence calculation method:
            - 'entropy' (DEFAULT): Entropy-based confidence (recommended)
            - 'max_prob': Max probability (ablation baseline)
        num_classes (int): Number of classes (only used for entropy method)
    
    Returns:
        confidence (np.ndarray): Confidence scores, shape (batch_size,), range [0, 1]
        predictions (np.ndarray): Predicted class indices, shape (batch_size,)
        probabilities (np.ndarray): Probability distribution, shape (batch_size, num_classes)
    """
    # Convert to torch if needed
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    
    # Compute probabilities
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    # Compute confidence based on method
    if method == 'entropy':
        confidence = get_entropy_based_confidence(logits, num_classes=num_classes)
    elif method == 'max_prob':
        confidence = get_max_probability_confidence(logits)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'entropy' or 'max_prob'.")
    
    return confidence, predictions, probabilities


def get_entropy_scores(logits, normalize=True):
    """
    Compute entropy scores for uncertainty quantification
    
    Formula:
        H(p) = -Σ(i=1 to C) p_i * log(p_i)
    
    Interpretation:
        - Lower entropy → model more certain
        - Higher entropy → model more uncertain
        - Max entropy = log(C) for uniform distribution
    
    Args:
        logits (torch.Tensor or np.ndarray): Model output logits, shape (batch_size, num_classes)
        normalize (bool): If True, normalize entropy to [0, 1] range
    
    Returns:
        entropy_scores (np.ndarray): Entropy values, shape (batch_size,)
    """
    # Convert to torch if needed
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    
    # Compute softmax probabilities
    softmax = torch.softmax(logits, dim=1)
    
    # Calculate entropy: H(p) = -Σ p_i * log(p_i)
    entropy_scores = -torch.sum(softmax * torch.log(softmax + 1e-8), dim=1)
    
    if normalize:
        # Normalize to [0, 1] range
        num_classes = logits.shape[1]
        max_entropy = np.log(num_classes)
        entropy_scores = entropy_scores / max_entropy
    
    return entropy_scores.cpu().numpy()


def analyze_confidence_distribution(confidence_scores):
    """
    Analyze distribution of confidence scores
    
    Args:
        confidence_scores (np.ndarray or torch.Tensor): Confidence scores, shape (batch_size,)
    
    Returns:
        stats (dict): Dictionary containing:
            - 'mean': Mean confidence
            - 'std': Standard deviation
            - 'min': Minimum confidence
            - 'max': Maximum confidence
            - 'median': Median confidence
            - 'q25': 25th percentile
            - 'q75': 75th percentile
    """
    if isinstance(confidence_scores, torch.Tensor):
        confidence_scores = confidence_scores.cpu().numpy()
    
    stats = {
        'mean': float(np.mean(confidence_scores)),
        'std': float(np.std(confidence_scores)),
        'min': float(np.min(confidence_scores)),
        'max': float(np.max(confidence_scores)),
        'median': float(np.median(confidence_scores)),
        'q25': float(np.percentile(confidence_scores, 25)),
        'q75': float(np.percentile(confidence_scores, 75))
    }
    
    return stats


def get_uncertain_predictions(confidence_scores, threshold=0.5):
    """
    Get indices of predictions with confidence below threshold
    
    Args:
        confidence_scores (np.ndarray or torch.Tensor): Confidence scores, shape (batch_size,)
        threshold (float): Confidence threshold (default 0.5, range [0, 1])
    
    Returns:
        uncertain_indices (np.ndarray): Indices of uncertain predictions
        uncertain_confidence (np.ndarray): Confidence values of uncertain predictions
    """
    if isinstance(confidence_scores, torch.Tensor):
        confidence_scores = confidence_scores.cpu().numpy()
    
    uncertain_indices = np.where(confidence_scores < threshold)[0]
    uncertain_confidence = confidence_scores[uncertain_indices]
    
    return uncertain_indices, uncertain_confidence
