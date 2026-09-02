"""
Stage 1a: Pretrain Face Model on FER2013

Trains FaceEmotionModel on the four FER2013 emotion classes:
    0 = angry
    1 = happy
    2 = neutral
    3 = sad

The best model is selected using validation Macro-F1 and
then reloaded and verified before IEMOCAP fine-tuning.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.face_model import FaceEmotionModel
from src.utils.iemocap_dataloader import FER2013Dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_epochs = 40
learning_rate = 1e-4
batch_size = 32
num_emotions = 4

patience = 7

fer2013_path = PROJECT_ROOT / "data" / "raw" / "FER2013"

checkpoint_path = (
    PROJECT_ROOT
    / "checkpoints"
    / "face_fer2013.pth"
)

checkpoint_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

emotion_names = {
    0: "angry",
    1: "happy",
    2: "neutral",
    3: "sad"
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STAGE 1a: PRETRAIN FACE MODEL ON FER2013")
print("=" * 70)

print(f"\nDevice:        {device}")
print(f"Epochs:        {num_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size:    {batch_size}")
print(f"Classes:       {num_emotions}")


# ============================================================
# LOAD FER2013
# ============================================================

print("\nLoading FER2013...")

full_train_dataset = FER2013Dataset(
    str(fer2013_path),
    split="train",
    target_size=48
)

test_dataset = FER2013Dataset(
    str(fer2013_path),
    split="test",
    target_size=48
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(
    0.90 * len(full_train_dataset)
)

val_size = (
    len(full_train_dataset)
    - train_size
)

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


print("\nDataset sizes:")
print(f"  Train: {len(train_dataset)}")
print(f"  Val:   {len(val_dataset)}")
print(f"  Test:  {len(test_dataset)}")


# ============================================================
# COMPUTE CLASS DISTRIBUTION
# ============================================================

print("\nComputing training class distribution...")

class_counts = np.zeros(
    num_emotions,
    dtype=np.int64
)

for _, labels in train_loader:

    for label in labels:

        class_counts[
            int(label.item())
        ] += 1


print("\nTraining distribution:")

for class_id in range(num_emotions):

    print(
        f"  {emotion_names[class_id]:8}: "
        f"{class_counts[class_id]}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_samples = class_counts.sum()

class_weights = (
    total_samples
    / (
        num_emotions
        * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("\nClass weights:")
print(class_weights)


# ============================================================
# INITIALISE MODEL
# ============================================================

face_model = FaceEmotionModel(
    num_emotions=num_emotions
).to(device)


print(
    f"\nModel parameters: "
    f"{sum(p.numel() for p in face_model.parameters()):,}"
)


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    face_model.parameters(),
    lr=learning_rate
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",          # because we monitor Macro-F1
    factor=0.5,
    patience=3
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    total_examples = 0

    all_labels = []
    all_predictions = []

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)

        labels = labels.to(
            device
        ).long()

        optimizer.zero_grad()

        # Raw logits
        outputs = model(
            images,
            return_embeddings=False
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size_current = images.size(0)

        running_loss += (
            loss.item()
            * batch_size_current
        )

        total_examples += (
            batch_size_current
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        if (batch_idx + 1) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1:4d}/"
                f"{len(loader):4d}"
            )

    epoch_loss = (
        running_loss
        / total_examples
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    total_examples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(
                device
            ).long()

            outputs = model(
                images,
                return_embeddings=False
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size_current = images.size(0)

            running_loss += (
                loss.item()
                * batch_size_current
            )

            total_examples += (
                batch_size_current
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    average_loss = (
        running_loss
        / total_examples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


best_val_f1 = -1.0
best_epoch = 0
patience_counter = 0


for epoch in range(
    1,
    num_epochs + 1
):

    print(
        f"\nEpoch {epoch}/{num_epochs}"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        train_loss,
        train_accuracy,
        train_f1
    ) = train_one_epoch(
        face_model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        val_f1,
        val_labels,
        val_predictions
    ) = evaluate(
        face_model,
        val_loader,
        criterion,
        device
    )

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    print(
        f"\nTrain Loss:     {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Train Macro-F1: {train_f1:.4f}"
    )

    print(
        f"\nVal Loss:       {val_loss:.4f}"
    )

    print(
        f"Val Accuracy:   {val_accuracy:.4f}"
    )

    print(
        f"Val Macro-F1:   {val_f1:.4f}"
    )

    print(
        f"Learning Rate:  {current_lr:.6f}"
    )

    # --------------------------------------------------------
    # LR scheduler monitors validation Macro-F1
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL ONLY
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            face_model.state_dict(),
            checkpoint_path
        )

        print(
            "\n✓ New best model saved"
        )

        print(
            f"  Validation Macro-F1: "
            f"{best_val_f1:.4f}"
        )

    else:

        patience_counter += 1

        print(
            f"\nNo improvement "
            f"({patience_counter}/{patience})"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= patience:

        print(
            f"\nEarly stopping at "
            f"epoch {epoch}."
        )

        break


# ============================================================
# IMPORTANT:
# DO NOT SAVE THE FINAL MODEL HERE.
# THE BEST CHECKPOINT IS ALREADY ON DISK.
# ============================================================


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation Macro-F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Best checkpoint: "
    f"{checkpoint_path}"
)


# ============================================================
# RELOAD BEST CHECKPOINT
# ============================================================

print("\nReloading best checkpoint...")

best_model = FaceEmotionModel(
    num_emotions=num_emotions
).to(device)

state_dict = torch.load(
    checkpoint_path,
    map_location=device
)

best_model.load_state_dict(
    state_dict
)

print("✓ Best checkpoint loaded successfully")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

(
    test_loss,
    test_accuracy,
    test_f1,
    test_labels,
    test_predictions
) = evaluate(
    best_model,
    test_loader,
    criterion,
    device
)


print("\n" + "=" * 70)
print("FINAL FER2013 TEST RESULTS")
print("=" * 70)

print(
    f"\nTest Loss:       "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy:   "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Macro-F1:   "
    f"{test_f1:.4f}"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

prediction_counts = Counter(
    test_predictions
)

print("\nPrediction Distribution:")

for class_id in range(
    num_emotions
):

    print(
        f"  {emotion_names[class_id]:8}: "
        f"{prediction_counts.get(class_id, 0)}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1, 2, 3]
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

unique_predictions = set(
    test_predictions
)

if len(unique_predictions) == 1:

    print(
        "\n⚠ WARNING:"
        "\nThe model predicts only ONE class."
        "\nDo NOT proceed to IEMOCAP fine-tuning."
    )

elif test_f1 <= 0.10:

    print(
        "\n⚠ WARNING:"
        "\nMacro-F1 is extremely low."
        "\nInspect training before fine-tuning."
    )

else:

    print(
        "\n✓ FER2013 pretraining produced "
        "a non-collapsed model."
    )

    print(
        "The checkpoint can now be considered "
        "for IEMOCAP fine-tuning."
    )