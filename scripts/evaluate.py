
import sys
from pathlib import Path


# Add the main project directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, random_split

from src.models.embedding_fusion import EmbeddingFusion
from src.models.face_model import FaceEmotionModel
from src.models.speech_model import SpeechEmotionModel
from src.utils.face_utils import forward_face_utterance
from src.utils.iemocap_multimodal_dataset import (
    IEMOCAPMultimodalDataset,
)

# CONFIGURATION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_emotions = 4

EMOTION_NAMES = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
EMOTION_ID_MAP = {v: k for k, v in EMOTION_NAMES.items()}

print("="*80)
print("EVALUATION: MULTIMODAL EMOTION RECOGNITION")
print("="*80)
print(f"\nDevice: {device}\n")

# LOAD MODELS

print("Loading trained models...")

# Speech model
speech_model = SpeechEmotionModel().to(device)
speech_ckpt = PROJECT_ROOT / "checkpoints" / "speech_iemocap.pth"
if speech_ckpt.exists():
    speech_model.load_state_dict(torch.load(speech_ckpt, map_location=device))
    print(f"  ✓ Speech model loaded: {speech_ckpt}")
else:
    print(f"  ✗ Speech model not found: {speech_ckpt}")
    exit(1)

# Face model
face_model = FaceEmotionModel().to(device)
face_ckpt = PROJECT_ROOT / "checkpoints" / "face_iemocap.pth"
if face_ckpt.exists():
    face_model.load_state_dict(torch.load(face_ckpt, map_location=device))
    print(f"  ✓ Face model loaded: {face_ckpt}")
else:
    print(f"  ✗ Face model not found: {face_ckpt}")
    exit(1)

# Dynamic fusion model
dynamic_fusion_model = EmbeddingFusion().to(device)
dynamic_fusion_ckpt = PROJECT_ROOT / "checkpoints" / "fusion_model_dynamic.pth"
if dynamic_fusion_ckpt.exists():
    dynamic_fusion_model.load_state_dict(torch.load(dynamic_fusion_ckpt, map_location=device))
    print(f"  ✓ Dynamic fusion model loaded: {dynamic_fusion_ckpt}")
else:
    print(f"  ✗ Dynamic fusion model not found: {dynamic_fusion_ckpt}")
    exit(1)

# Fixed 50/50 fusion model (REQUIRED for this experiment)
fixed_fusion_model = EmbeddingFusion().to(device)
fixed_fusion_ckpt = PROJECT_ROOT / "checkpoints" / "fusion_model_equal.pth"
if not fixed_fusion_ckpt.exists():
    raise FileNotFoundError(
        f"\n\nFIXED 50/50 FUSION CHECKPOINT NOT FOUND\n"
        f"Expected path: {fixed_fusion_ckpt}\n\n"
        f"Please train the fixed 50/50 model first:\n"
        f"  1. Edit scripts/train_multimodal_fusion.py: fusion_mode = \"equal\"\n"
        f"  2. Run: python scripts/train_multimodal_fusion.py\n\n"
    )
fixed_fusion_model.load_state_dict(torch.load(fixed_fusion_ckpt, map_location=device))
print(f"  ✓ Fixed 50/50 fusion model loaded: {fixed_fusion_ckpt}\n")

# ===== SET TO EVAL MODE =====
speech_model.eval()
face_model.eval()
dynamic_fusion_model.eval()
fixed_fusion_model.eval()

print("Evaluating 4 models: Speech, Face, Fixed 50/50 Fusion, Dynamic Entropy Fusion\n")

# LOAD TEST DATA

print("Loading multimodal dataset...")

# Load full dataset (this automatically handles MFCC extraction and face loading)
full_dataset = IEMOCAPMultimodalDataset(
    csv_path='data/processed/metadata/iemocap_harmonized.csv',
    audio_dir='data/raw/IEMOCAP/IEMOCAP_full_release',
    face_dir='data/processed/IEMOCAP_faces',
    n_mfcc=13,
    target_length=128,
    face_size=48,
    training=False  # No augmentation during evaluation
)

print(f"  Full dataset size: {len(full_dataset)}\n")

# Create 80/20 train/test split (matching training split)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  Train set: {len(train_dataset)}")
print(f"  Test set: {len(test_dataset)}\n")

# Create test dataloader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# HELPER FUNCTION: CONFIDENCE-BASED WEIGHTING

def apply_confidence_weights(speech_embeddings, face_embeddings,
                            speech_logits, face_logits, num_emotions, fusion_mode="dynamic"):
    """
    Apply modality weighting to embeddings.
    
    fusion_mode:
        "dynamic": entropy-based confidence weighting
        "equal": fixed 0.5/0.5 weighting (baseline comparison)
    """
    
    if fusion_mode == "equal":
        # FIXED 50/50 WEIGHTING BASELINE 
        batch_size = speech_embeddings.size(0)
        speech_weight = 0.5 * torch.ones(batch_size, 1, dtype=speech_embeddings.dtype, device=device)
        face_weight = 0.5 * torch.ones(batch_size, 1, dtype=face_embeddings.dtype, device=device)
    else:
        # DYNAMIC ENTROPY-BASED WEIGHTING 
        # Compute softmax probabilities
        speech_probs = torch.softmax(speech_logits, dim=1)
        face_probs = torch.softmax(face_logits, dim=1)
        
        # Compute entropy: H(p) = -Σ p_i * log(p_i)
        speech_entropy = -torch.sum(speech_probs * torch.log(speech_probs + 1e-8), dim=1)
        face_entropy = -torch.sum(face_probs * torch.log(face_probs + 1e-8), dim=1)
        
        # Normalize entropy to [0, 1]: c = 1 - H(p) / log(C)
        max_entropy = np.log(num_emotions)
        speech_confidence = 1 - (speech_entropy / max_entropy)
        face_confidence = 1 - (face_entropy / max_entropy)
        
        # Normalize weights: w = c / (c_s + c_f + epsilon)
        epsilon = 1e-8
        weight_sum = speech_confidence + face_confidence + epsilon
        speech_weight = speech_confidence / weight_sum
        face_weight = face_confidence / weight_sum
        speech_weight = speech_weight.unsqueeze(1)
        face_weight = face_weight.unsqueeze(1)
    
    # Apply weights to embeddings
    weighted_speech = speech_embeddings * speech_weight
    weighted_face = face_embeddings * face_weight
    
    return weighted_speech, weighted_face

# EVALUATION LOOP

print("EVALUATING ON TEST SET")

speech_preds = []
speech_probs = []
face_preds = []
face_probs = []
dynamic_fusion_preds = []
dynamic_fusion_probs = []
fixed_fusion_preds = []
fixed_fusion_probs = []
true_labels = []

with torch.no_grad():
    for batch_idx, (
        speech_mfcc,
        face_frames,
        frame_mask,
        labels
    ) in enumerate(test_loader):

        speech_mfcc = speech_mfcc.to(device)
        face_frames = face_frames.to(device)
        frame_mask = frame_mask.to(device)
        labels = labels.to(device)
        
        
        #  SPEECH MODEL 
        speech_embeddings = speech_model(speech_mfcc, return_embeddings=True)
        speech_logits = speech_model(speech_mfcc, return_embeddings=False)
        speech_pred = torch.argmax(speech_logits, dim=1)
        speech_preds.extend(speech_pred.cpu().numpy())
        speech_probs.extend(torch.softmax(speech_logits, dim=1).cpu().numpy())
        
        #  FACE MODEL 
        face_embeddings, face_logits = forward_face_utterance(
            model=face_model,
            frames=face_frames,
            frame_mask=frame_mask,
        )
        face_pred = torch.argmax(face_logits, dim=1)
        face_preds.extend(face_pred.cpu().numpy())
        face_probs.extend(torch.softmax(face_logits, dim=1).cpu().numpy())
        
        # DYNAMIC ENTROPY FUSION MODEL 
        # Apply entropy-based confidence weighting
        weighted_speech_dyn, weighted_face_dyn = apply_confidence_weights(
            speech_embeddings, face_embeddings,
            speech_logits, face_logits,
            num_emotions,
            fusion_mode="dynamic"
        )
        
        # Dynamic fusion forward pass
        fusion_output_dyn = dynamic_fusion_model(weighted_speech_dyn, weighted_face_dyn)
        fusion_logits_dyn = fusion_output_dyn['logits']
        fusion_pred_dyn = torch.argmax(fusion_logits_dyn, dim=1)
        dynamic_fusion_preds.extend(fusion_pred_dyn.cpu().numpy())
        dynamic_fusion_probs.extend(torch.softmax(fusion_logits_dyn, dim=1).cpu().numpy())
        
    
        # Apply fixed 0.5/0.5 weighting
        weighted_speech_eq, weighted_face_eq = apply_confidence_weights(
            speech_embeddings, face_embeddings,
            speech_logits, face_logits,
            num_emotions,
            fusion_mode="equal"
        )
        
        # Fixed fusion forward pass
        fusion_output_eq = fixed_fusion_model(weighted_speech_eq, weighted_face_eq)
        fusion_logits_eq = fusion_output_eq['logits']
        fusion_pred_eq = torch.argmax(fusion_logits_eq, dim=1)
        fixed_fusion_preds.extend(fusion_pred_eq.cpu().numpy())
        fixed_fusion_probs.extend(torch.softmax(fusion_logits_eq, dim=1).cpu().numpy())
        
        # Store true labels
        true_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

print(f"\n✓ Evaluation complete on {len(true_labels)} samples\n")

# VALIDATION CHECKS

print("="*80)
print("VALIDATION CHECKS")
print("="*80)

# Convert to numpy arrays
speech_probs = np.array(speech_probs)
face_probs = np.array(face_probs)
dynamic_fusion_probs = np.array(dynamic_fusion_probs)
fixed_fusion_probs = np.array(fixed_fusion_probs)
true_labels_arr = np.array(true_labels)
speech_preds_arr = np.array(speech_preds)
face_preds_arr = np.array(face_preds)
dynamic_fusion_preds_arr = np.array(dynamic_fusion_preds)
fixed_fusion_preds_arr = np.array(fixed_fusion_preds)

print(f"Number of test samples:                {len(true_labels_arr)}")
print(f"Number of true labels:                 {len(true_labels_arr)}")
print(f"Number of speech predictions:          {len(speech_preds_arr)}")
print(f"Number of face predictions:            {len(face_preds_arr)}")
print(f"Number of dynamic fusion predictions:  {len(dynamic_fusion_preds_arr)}")
print(f"Number of fixed fusion predictions:    {len(fixed_fusion_preds_arr)}")
print(f"\nSpeech probability array shape:        {speech_probs.shape}")
print(f"Face probability array shape:          {face_probs.shape}")
print(f"Dynamic fusion probability shape:      {dynamic_fusion_probs.shape}")
print(f"Fixed fusion probability shape:        {fixed_fusion_probs.shape}")
print(f"\nSpeech probs sum per sample (first 3):  {speech_probs[:3].sum(axis=1)}")
print(f"Face probs sum per sample (first 3):    {face_probs[:3].sum(axis=1)}")
print(f"Dynamic fusion probs sum (first 3):     {dynamic_fusion_probs[:3].sum(axis=1)}")
print(f"Fixed fusion probs sum (first 3):       {fixed_fusion_probs[:3].sum(axis=1)}")
print(f"\nEmotion class order: Angry(0), Happy(1), Neutral(2), Sad(3)")
print(f"True labels unique values:             {np.unique(true_labels_arr)}")
print(f"All models evaluated on same true labels: CONFIRMED\n")

# HELPER FUNCTIONS FOR NEW METRICS

def compute_specificity_per_class(y_true, y_pred, num_classes):
    """Compute specificity (TNR) per class using one-vs-rest approach"""
    specificities = []
    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        
        # True negatives: correctly predicted as not this class
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        # False positives: incorrectly predicted as this class
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    return specificities

def compute_roc_auc_per_class(y_true, y_probs, num_classes):
    """Compute ROC-AUC per class using one-vs-rest approach"""
    roc_aucs = []
    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        y_probs_c = y_probs[:, c]
        try:
            roc_auc = roc_auc_score(y_true_bin, y_probs_c)
            roc_aucs.append(roc_auc)
        except:
            roc_aucs.append(np.nan)
    
    return roc_aucs

def compute_pr_auc_per_class(y_true, y_probs, num_classes):
    """Compute PR-AUC / Average Precision per class using one-vs-rest approach"""
    pr_aucs = []
    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        y_probs_c = y_probs[:, c]
        
        precision, recall, _ = precision_recall_curve(y_true_bin, y_probs_c)
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
    
    return pr_aucs

# COMPUTE METRICS

def compute_metrics(y_true, y_pred, y_probs, model_name, num_classes=4):
    """Compute evaluation metrics including new metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Compute specificity per class
    specificities = compute_specificity_per_class(y_true, y_pred, num_classes)
    specificity_macro = np.mean(specificities)
    
    # Compute ROC-AUC per class
    roc_aucs = compute_roc_auc_per_class(y_true, y_probs, num_classes)
    roc_auc_macro = np.nanmean(roc_aucs)
    
    # Compute weighted ROC-AUC
    class_support = np.array([(y_true == c).sum() for c in range(num_classes)])
    weights = class_support / class_support.sum()
    roc_auc_weighted = np.nansum(np.array(roc_aucs) * weights)
    
    # Compute PR-AUC per class
    pr_aucs = compute_pr_auc_per_class(y_true, y_probs, num_classes)
    pr_auc_macro = np.mean(pr_aucs)
    pr_auc_weighted = np.sum(np.array(pr_aucs) * weights)
    
    print(f"{model_name:20} Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"{'':20} Specificity (macro): {specificity_macro:.4f}")
    print(f"{'':20} ROC-AUC (macro): {roc_auc_macro:.4f} | ROC-AUC (weighted): {roc_auc_weighted:.4f}")
    print(f"{'':20} PR-AUC (macro): {pr_auc_macro:.4f} | PR-AUC (weighted): {pr_auc_weighted:.4f}\n")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity_macro': specificity_macro,
        'specificity_angry': specificities[0],
        'specificity_happy': specificities[1],
        'specificity_neutral': specificities[2],
        'specificity_sad': specificities[3],
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'roc_auc_angry': roc_aucs[0],
        'roc_auc_happy': roc_aucs[1],
        'roc_auc_neutral': roc_aucs[2],
        'roc_auc_sad': roc_aucs[3],
        'pr_auc_macro': pr_auc_macro,
        'pr_auc_weighted': pr_auc_weighted,
        'pr_auc_angry': pr_aucs[0],
        'pr_auc_happy': pr_aucs[1],
        'pr_auc_neutral': pr_aucs[2],
        'pr_auc_sad': pr_aucs[3],
    }

# Compute for each model
print("="*80)
print("METRICS")
print("="*80)

# Sanity check: verify fixed fusion predictions were collected
assert len(fixed_fusion_preds) == len(true_labels), \
    f"Fixed fusion predictions count mismatch: {len(fixed_fusion_preds)} vs {len(true_labels)} labels"

speech_metrics = compute_metrics(true_labels_arr, speech_preds_arr, speech_probs, 'Speech')
face_metrics = compute_metrics(true_labels_arr, face_preds_arr, face_probs, 'Face')
fixed_fusion_metrics = compute_metrics(true_labels_arr, fixed_fusion_preds_arr, fixed_fusion_probs, 'Fixed 50/50 Fusion')
dynamic_fusion_metrics = compute_metrics(true_labels_arr, dynamic_fusion_preds_arr, dynamic_fusion_probs, 'Dynamic Entropy Fusion')

# SAVE METRICS TO CSV

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80 + "\n")

# Create metrics dataframe
metrics_df = pd.DataFrame([
    speech_metrics,
    face_metrics,
    fixed_fusion_metrics,
    dynamic_fusion_metrics
])
metrics_csv_path = PROJECT_ROOT / "results" / "evaluation_metrics.csv"
metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"✓ Metrics saved to: {metrics_csv_path}")

# CONFUSION MATRIX

# Compute confusion matrices
speech_cm = confusion_matrix(true_labels, speech_preds, labels=range(num_emotions))
face_cm = confusion_matrix(true_labels, face_preds, labels=range(num_emotions))
fixed_fusion_cm = confusion_matrix(true_labels, fixed_fusion_preds, labels=range(num_emotions))
dynamic_fusion_cm = confusion_matrix(true_labels, dynamic_fusion_preds, labels=range(num_emotions))

# Create visualization (4 plots)
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
axes = list(axes)

emotion_labels = [EMOTION_NAMES[i].capitalize() for i in range(num_emotions)]

plot_data = [
    (speech_cm, axes[0], 'Speech Model'),
    (face_cm, axes[1], 'Face Model'),
    (fixed_fusion_cm, axes[2], 'Fixed 50/50 Fusion'),
    (dynamic_fusion_cm, axes[3], 'Dynamic Entropy Fusion'),
]

for cm, ax, title in plot_data:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                cbar=False)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
confusion_matrix_path = PROJECT_ROOT / "results" / "confusion_matrices.png"
confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {confusion_matrix_path}")
plt.close()

# MODEL PERFORMANCE COMPARISON (Grouped Bar Chart)

print("\nGenerating visualizations...")

fig, ax = plt.subplots(figsize=(12, 6))

models = ['Speech', 'Face', 'Fixed 50/50 Fusion', 'Dynamic Entropy Fusion']
x = np.arange(len(models))
width = 0.2

metrics_to_plot = {
    'Accuracy': [speech_metrics['accuracy'], face_metrics['accuracy'], 
                 fixed_fusion_metrics['accuracy'], dynamic_fusion_metrics['accuracy']],
    'Weighted Precision': [speech_metrics['precision'], face_metrics['precision'],
                           fixed_fusion_metrics['precision'], dynamic_fusion_metrics['precision']],
    'Weighted Recall': [speech_metrics['recall'], face_metrics['recall'],
                        fixed_fusion_metrics['recall'], dynamic_fusion_metrics['recall']],
    'Weighted F1-Score': [speech_metrics['f1'], face_metrics['f1'],
                          fixed_fusion_metrics['f1'], dynamic_fusion_metrics['f1']],
}

for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
    ax.bar(x + i*width, values, width, label=metric_name)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - Core Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models)
ax.legend(loc='lower right')
ax.set_ylim([0.4, 0.8])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
comparison_path = PROJECT_ROOT / "results" / "model_comparison.png"
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"✓ Model comparison saved to: {comparison_path}")
plt.close()

# ROC CURVES - Fixed 50/50 Fusion

fig, ax = plt.subplots(figsize=(10, 8))

emotion_names_list = ['Angry', 'Happy', 'Neutral', 'Sad']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# Compute ROC curves for fixed fusion
fixed_roc_data = []
for c in range(num_emotions):
    y_true_bin = (true_labels_arr == c).astype(int)
    y_probs_c = fixed_fusion_probs[:, c]
    fpr, tpr, _ = roc_curve(y_true_bin, y_probs_c)
    roc_auc = roc_auc_score(y_true_bin, y_probs_c)
    fixed_roc_data.append((fpr, tpr, roc_auc))
    ax.plot(fpr, tpr, label=f'{emotion_names_list[c]} (AUC = {roc_auc:.3f})',
            color=colors[c], linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Fixed 50/50 Fusion Model', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
roc_fixed_path = PROJECT_ROOT / "results" / "roc_curves_fixed_fusion.png"
plt.savefig(roc_fixed_path, dpi=300, bbox_inches='tight')
print(f"✓ ROC curves (Fixed 50/50) saved to: {roc_fixed_path}")
plt.close()

# ROC CURVES - Dynamic Entropy Fusion

fig, ax = plt.subplots(figsize=(10, 8))

# Compute ROC curves for dynamic fusion
dynamic_roc_data = []
for c in range(num_emotions):
    y_true_bin = (true_labels_arr == c).astype(int)
    y_probs_c = dynamic_fusion_probs[:, c]
    fpr, tpr, _ = roc_curve(y_true_bin, y_probs_c)
    roc_auc = roc_auc_score(y_true_bin, y_probs_c)
    dynamic_roc_data.append((fpr, tpr, roc_auc))
    ax.plot(fpr, tpr, label=f'{emotion_names_list[c]} (AUC = {roc_auc:.3f})',
            color=colors[c], linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Dynamic Entropy Fusion Model', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
roc_dynamic_path = PROJECT_ROOT / "results" / "roc_curves_dynamic_fusion.png"
plt.savefig(roc_dynamic_path, dpi=300, bbox_inches='tight')
print(f"✓ ROC curves (Dynamic Entropy) saved to: {roc_dynamic_path}")
plt.close()

# PRECISION-RECALL CURVES - Fixed 50/50 Fusion

fig, ax = plt.subplots(figsize=(10, 8))

# Compute PR curves for fixed fusion
for c in range(num_emotions):
    y_true_bin = (true_labels_arr == c).astype(int)
    y_probs_c = fixed_fusion_probs[:, c]
    precision, recall, _ = precision_recall_curve(y_true_bin, y_probs_c)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, label=f'{emotion_names_list[c]} (AP = {pr_auc:.3f})',
            color=colors[c], linewidth=2)

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - Fixed 50/50 Fusion Model', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
pr_fixed_path = PROJECT_ROOT / "results" / "pr_curves_fixed_fusion.png"
plt.savefig(pr_fixed_path, dpi=300, bbox_inches='tight')
print(f"✓ PR curves (Fixed 50/50) saved to: {pr_fixed_path}")
plt.close()

# PRECISION-RECALL CURVES - Dynamic Entropy Fusion

fig, ax = plt.subplots(figsize=(10, 8))

# Compute PR curves for dynamic fusion
for c in range(num_emotions):
    y_true_bin = (true_labels_arr == c).astype(int)
    y_probs_c = dynamic_fusion_probs[:, c]
    precision, recall, _ = precision_recall_curve(y_true_bin, y_probs_c)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, label=f'{emotion_names_list[c]} (AP = {pr_auc:.3f})',
            color=colors[c], linewidth=2)

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - Dynamic Entropy Fusion Model', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
pr_dynamic_path = PROJECT_ROOT / "results" / "pr_curves_dynamic_fusion.png"
plt.savefig(pr_dynamic_path, dpi=300, bbox_inches='tight')
print(f"✓ PR curves (Dynamic Entropy) saved to: {pr_dynamic_path}")
plt.close()

print(f"\n✓ All visualizations generated.\n")

# CLASSIFICATION REPORT

report_path = PROJECT_ROOT / "results" / "classification_report.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLASSIFICATION REPORT - TEST SET\n")
    f.write("="*80 + "\n\n")
    
    # Speech model
    f.write("SPEECH MODEL\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(true_labels_arr, speech_preds_arr,
                                  target_names=emotion_labels, digits=4))
    f.write("\n\n")
    
    # Face model
    f.write("FACE MODEL\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(true_labels_arr, face_preds_arr,
                                  target_names=emotion_labels, digits=4))
    f.write("\n\n")
    
    # Fixed 50/50 fusion model
    f.write("FIXED 50/50 FUSION MODEL\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(true_labels_arr, fixed_fusion_preds_arr,
                                  target_names=emotion_labels, digits=4))
    f.write("\n\n")
    
    # Dynamic entropy fusion model
    f.write("DYNAMIC ENTROPY FUSION MODEL\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(true_labels_arr, dynamic_fusion_preds_arr,
                                  target_names=emotion_labels, digits=4))
    f.write("\n")

print(f"✓ Classification report saved to: {report_path}")

# SUMMARY

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTest samples: {len(true_labels_arr)}")
print(f"\nSpeech model accuracy:           {speech_metrics['accuracy']:.4f}")
print(f"Face model accuracy:             {face_metrics['accuracy']:.4f}")
print(f"Fixed 50/50 Fusion accuracy:     {fixed_fusion_metrics['accuracy']:.4f}")
print(f"Dynamic Entropy Fusion accuracy:  {dynamic_fusion_metrics['accuracy']:.4f}")

print(f"\nComparison - Fixed 50/50 vs Individual Models:")
print(f"  vs Speech: +{(fixed_fusion_metrics['accuracy'] - speech_metrics['accuracy'])*100:.2f}%")
print(f"  vs Face:   +{(fixed_fusion_metrics['accuracy'] - face_metrics['accuracy'])*100:.2f}%")
print(f"\nComparison - Dynamic Entropy vs Individual Models:")
print(f"  vs Speech: +{(dynamic_fusion_metrics['accuracy'] - speech_metrics['accuracy'])*100:.2f}%")
print(f"  vs Face:   +{(dynamic_fusion_metrics['accuracy'] - face_metrics['accuracy'])*100:.2f}%")
print(f"\nComparison - Dynamic vs Fixed 50/50:")
print(f"  Improvement: +{(dynamic_fusion_metrics['accuracy'] - fixed_fusion_metrics['accuracy'])*100:.2f}%")

print("\n" + "="*80)
print("✓ EVALUATION COMPLETE")
print("="*80 + "\n")

print("Results saved to:")
print(f"  - Metrics CSV: {metrics_csv_path}")
print(f"  - Confusion matrices: {confusion_matrix_path}")
print(f"  - Classification report: {report_path}")
print(f"  - Model comparison: {comparison_path}")
print(f"  - ROC curves (Fixed 50/50): {roc_fixed_path}")
print(f"  - ROC curves (Dynamic Entropy): {roc_dynamic_path}")
print(f"  - PR curves (Fixed 50/50): {pr_fixed_path}")
print(f"  - PR curves (Dynamic Entropy): {pr_dynamic_path}")
