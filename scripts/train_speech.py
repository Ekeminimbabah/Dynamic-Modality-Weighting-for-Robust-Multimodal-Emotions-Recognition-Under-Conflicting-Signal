import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.speech_model import SpeechEmotionModel
from src.utils.iemocap_dataloader import get_dataloaders
#from src.utils.audio_utils import extract_mfcc

# ============================================================
# CONFIGURATION
# ============================================================
# Training hyperparameters for speech emotion recognition on IEMOCAP dataset.
# IEMOCAP is imbalanced (more neutral/happy than angry/sad), so class weights are applied.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 50
learning_rate = 0.001
batch_size = 32
num_emotions = 4

print("="*70)
print("STAGE 1b: TRAIN SPEECH MODEL ON IEMOCAP")
print("="*70)
print(f"\nDevice: {device}")
print(f"Epochs: {num_epochs}")
print(f"Learning Rate: {learning_rate}\n")

# ============================================================
# COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================
# IEMOCAP has uneven emotion distribution. Inverse frequency weighting ensures
# that minority classes (angry, sad) receive higher loss weight, preventing
# the model from ignoring these emotions during training.

print("Computing class weights...")
csv_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "iemocap_harmonized.csv"
df = pd.read_csv(csv_path)

# Count class distribution in full dataset
class_counts = df['label_id'].value_counts().sort_index().values
print(f"  Class distribution: {class_counts}")

# Compute weights: inverse frequency
# Formula: weight = total_samples / (num_classes * samples_in_class)
# Higher weight for underrepresented emotions
total_samples = len(df)
class_weights = total_samples / (num_emotions * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"  Class weights: {class_weights}\n")

# ============================================================
# INITIALIZE MODEL
# ============================================================
# Load SpeechEmotionModel architecture and optimizer.
# Adam optimizer with ReduceLROnPlateau scheduler: reduces LR if validation metric plateaus.

speech_model = SpeechEmotionModel(num_emotions=num_emotions).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(speech_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f"✓ Model initialized")
print(f"  Parameters: {sum(p.numel() for p in speech_model.parameters()):,}\n")

# ============================================================
# LOAD DATA
# ============================================================
# Load IEMOCAP MFCC features with same train/val/test split as facial model.
# Consistent partitioning ensures fair multimodal comparison.

print("\n[1] Calling get_dataloaders...", flush=True)


loaders = get_dataloaders(
    batch_size=batch_size,
    num_workers=0
)

print("[2] get_dataloaders returned successfully", flush=True)
print("[3] Loader keys:", loaders.keys(), flush=True)


train_speech = loaders["train_speech"]
val_speech = loaders["val_speech"]

features, labels = next(iter(train_speech))
print("Training feature shape:", features.shape)
print("Training label shape:", labels.shape)

print(f"[4] Training batches: {len(train_speech)}", flush=True)
print(f"[5] Validation batches: {len(val_speech)}", flush=True)

print("[6] Testing first training batch...", flush=True)

X_test, y_test = next(iter(train_speech))

print("[7] First batch loaded successfully", flush=True)
print("    X shape:", X_test.shape, flush=True)
print("    y shape:", y_test.shape, flush=True)
print("    Labels:", torch.unique(y_test), flush=True)

print("=" * 70)
print("TRAINING IS STARTING NOW", flush=True)
print("=" * 70)


# TRAINING LOOP
best_val_loss = float("inf")
patience = 5
patience_counter = 0
min_delta = 0.001

for epoch in range(num_epochs):

    # ================= TRAINING =================
    speech_model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (X, y) in enumerate(train_speech):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = speech_model(X, return_embeddings=False)
        loss = criterion(outputs, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            speech_model.parameters(),
            max_norm=1.0
        )
        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == y).sum().item()
        train_total += y.size(0)

        if (batch_idx + 1) % 10 == 0:
            running_loss = train_loss / (batch_idx + 1)
            running_accuracy = 100 * train_correct / train_total

            print(
                f"  Epoch {epoch + 1:2d}/{num_epochs} | "
                f"Batch {batch_idx + 1:4d}/{len(train_speech):4d} | "
                f"Loss: {running_loss:.4f} | "
                f"Accuracy: {running_accuracy:.2f}%"
            )

    epoch_train_loss = train_loss / len(train_speech)
    epoch_train_accuracy = 100 * train_correct / train_total

    # ======== VALIDATION =========
    speech_model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_val, y_val in val_speech:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            val_outputs = speech_model(
                X_val,
                return_embeddings=False
            )

            loss = criterion(val_outputs, y_val)
            val_loss += loss.item()

            val_predictions = val_outputs.argmax(dim=1)
            val_correct += (
                val_predictions == y_val
            ).sum().item()

            val_total += y_val.size(0)

    epoch_val_loss = val_loss / len(val_speech)
    epoch_val_accuracy = 100 * val_correct / val_total

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\n✓ Epoch {epoch + 1:2d}/{num_epochs} Complete\n"
        f"  Train Loss: {epoch_train_loss:.4f} | "
        f"Train Accuracy: {epoch_train_accuracy:.2f}%\n"
        f"  Val Loss: {epoch_val_loss:.4f} | "
        f"Val Accuracy: {epoch_val_accuracy:.2f}%\n"
        f"  Learning Rate: {current_lr:.6f}\n"
    )

    scheduler.step(epoch_val_loss)

    # ======== EARLY STOPPING ========
    if epoch_val_loss < best_val_loss - min_delta:
        best_val_loss = epoch_val_loss
        patience_counter = 0

        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "speech_iemocap.pth"
        )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            speech_model.state_dict(),
            checkpoint_path
        )

        print(
            f"  ✓ Best model saved "
            f"(val loss: {epoch_val_loss:.4f})\n"
        )

    else:
        patience_counter += 1

        print(
            f"  No validation improvement "
            f"({patience_counter}/{patience})\n"
        )

        if patience_counter >= patience:
            print(
                f"  Early stopping at epoch "
                f"{epoch + 1}\n"
            )
            break

# TRAINING COMPLETE

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

checkpoint_path = (
    PROJECT_ROOT
    / "checkpoints"
    / "speech_iemocap.pth"
)

print(f"\n✓ Best speech model retained at: {checkpoint_path}")
print(f"✓ Best validation loss: {best_val_loss:.4f}")
print("✓ Ready for multimodal fusion training")
print("\n✓ Done!")

speech_model.load_state_dict(
    torch.load(
        PROJECT_ROOT / "checkpoints" / "speech_iemocap.pth",
        map_location=device,
    )
)

speech_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for X_val, y_val in val_speech:
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        outputs = speech_model(
            X_val,
            return_embeddings=False,
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == y_val
        ).sum().item()

        total += y_val.size(0)

print(
    "Reloaded checkpoint validation accuracy:",
    correct / total
)