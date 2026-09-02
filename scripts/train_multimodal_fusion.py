
import sys
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.speech_model import SpeechEmotionModel
from src.models.face_model import FaceEmotionModel
from src.models.embedding_fusion import EmbeddingFusion
from src.utils.confidence import get_entropy_based_confidence
from src.utils.face_utils import (
    get_face_classifier,
    extract_face_embeddings,
    forward_face_utterance,
)
from src.utils.audio_utils import extract_mfcc
from src.utils.iemocap_multimodal_dataset import (
    IEMOCAPMultimodalDataset,
)

# CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
learning_rate = 0.001
batch_size = 32
num_emotions = 4
random_seed = 42
fusion_mode = "equal"  # "dynamic" (entropy-based) or "equal" (0.5/0.5)


print("STAGE 3: TRAIN MULTIMODAL FUSION WITH CONFIDENCE WEIGHTING")
print(f"\nDevice: {device}")
print(f"Epochs: {num_epochs}")
print(f"Learning Rate: {learning_rate}\n")

# UTTERANCE-LEVEL MULTIMODAL DATASET

class IEMOCAPMultimodalDataset(Dataset):
    """
    Pair one speech MFCC tensor with up to three face frames from the
    same IEMOCAP utterance.
    """

    FRAME_NAMES = ("start.png", "middle.png", "end.png")

    EMOTION_TO_ID = {
        "angry": 0,
        "happy": 1,
        "neutral": 2,
        "sad": 3,
    }

    POSSIBLE_ID_COLUMNS = (
        "utterance_id",
        "utterance",
        "utt_id",
        "id",
        "filename",
        "file_name",
        "audio_filename",
        "wav_file",
        "audio_path",
        "path",
    )

    POSSIBLE_AUDIO_COLUMNS = (
        "audio_path",
        "wav_path",
        "file_path",
        "path",
        "audio_file",
        "wav_file",
        "filename",
        "file_name",
    )

    POSSIBLE_LABEL_COLUMNS = (
        "label_id",
        "emotion_id",
        "class_id",
        "label",
        "emotion",
    )

    def __init__(
        self,
        csv_path,
        audio_dir,
        face_dir,
        n_mfcc=13,
        target_length=128,
        face_size=48,
        training=False,
    ):
        print("Loading IEMOCAP multimodal data")
        print("  Pairing: speech MFCC + utterance-level face frames\n")

        self.csv_path = self._resolve_project_path(csv_path)
        self.audio_dir = self._resolve_project_path(audio_dir)
        self.face_dir = self._resolve_project_path(face_dir)

        self.n_mfcc = n_mfcc
        self.target_length = target_length
        self.face_size = face_size
        self.training = training

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found:\n{self.csv_path}"
            )

        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"IEMOCAP audio directory not found:\n{self.audio_dir}"
            )

        if not self.face_dir.exists():
            raise FileNotFoundError(
                f"IEMOCAP face directory not found:\n{self.face_dir}"
            )

        self.metadata = pd.read_csv(self.csv_path)

        self.id_column = self._find_column(
            self.POSSIBLE_ID_COLUMNS,
            required=True,
            purpose="utterance identifier",
        )

        self.audio_column = self._find_column(
            self.POSSIBLE_AUDIO_COLUMNS,
            required=False,
            purpose="audio path",
        )

        self.label_column = self._find_column(
            self.POSSIBLE_LABEL_COLUMNS,
            required=True,
            purpose="emotion label",
        )

        self.face_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((face_size, face_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.face_lookup = self._build_face_lookup()
        self.audio_lookup = self._build_audio_lookup()

        print(f"  Metadata rows: {len(self.metadata)}")
        print(f"  Face utterance folders found: {len(self.face_lookup)}")
        print(f"  Audio files indexed: {len(self.audio_lookup)}")
        print(f"  ID column: {self.id_column}")
        print(f"  Audio column: {self.audio_column or 'not present; using audio index'}")
        print(f"  Label column: {self.label_column}")

        self.samples = []
        unmatched_face_ids = []
        unmatched_audio_ids = []

        for _, row in self.metadata.iterrows():
            utterance_id = self._normalise_utterance_id(
                row[self.id_column]
            )

            if not utterance_id:
                continue

            face_folder = self.face_lookup.get(utterance_id)

            if face_folder is None:
                unmatched_face_ids.append(utterance_id)
                continue

            audio_path = self._resolve_audio_path(
                row=row,
                utterance_id=utterance_id,
            )

            if audio_path is None:
                unmatched_audio_ids.append(utterance_id)
                continue

            label = self._normalise_label(
                row[self.label_column]
            )

            if label is None:
                continue

            self.samples.append(
                {
                    "utterance_id": utterance_id,
                    "audio_path": audio_path,
                    "face_folder": face_folder,
                    "label": label,
                }
            )

        self.labels = [
            sample["label"]
            for sample in self.samples
        ]

        print(f"\n✓ Loaded {len(self.samples)} paired samples from IEMOCAP")

        if self.samples:
            class_counts = np.bincount(
                np.asarray(self.labels, dtype=np.int64),
                minlength=4,
            )

            print("  Emotion distribution:")
            for emotion_name, label_id in self.EMOTION_TO_ID.items():
                print(
                    f"    {emotion_name:<8}: "
                    f"{class_counts[label_id]}"
                )
        else:
            print("  Example face IDs:", list(self.face_lookup)[:5])
            print("  Example audio IDs:", list(self.audio_lookup)[:5])
            print("  Example metadata IDs:")
            for value in self.metadata[self.id_column].head(5):
                print(
                    "   ",
                    self._normalise_utterance_id(value),
                )

            raise RuntimeError(
                "No multimodal samples could be paired.\n"
                "The diagnostic IDs printed above show how the metadata, "
                "audio files and face folders are being interpreted."
            )

        if unmatched_face_ids:
            print(
                f"  Metadata rows without face match: "
                f"{len(set(unmatched_face_ids))}"
            )

        if unmatched_audio_ids:
            print(
                f"  Metadata rows without audio match: "
                f"{len(set(unmatched_audio_ids))}"
            )

        print()

    @staticmethod
    def _resolve_project_path(path_value):
        path = Path(path_value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    def _find_column(self, candidates, required, purpose):
        column_lookup = {
            str(column).strip().lower(): column
            for column in self.metadata.columns
        }

        for candidate in candidates:
            match = column_lookup.get(candidate.lower())

            if match is not None:
                return match

        if required:
            raise KeyError(
                f"Could not find a column for {purpose}.\n"
                f"Available columns: {self.metadata.columns.tolist()}"
            )

        return None

    @staticmethod
    def _normalise_utterance_id(value):
        if pd.isna(value):
            return ""

        value = str(value).strip().replace("\\", "/")
        value = value.split("/")[-1]

        known_extensions = (
            ".wav",
            ".WAV",
            ".png",
            ".jpg",
            ".jpeg",
            ".avi",
            ".mp4",
        )

        for extension in known_extensions:
            if value.endswith(extension):
                value = value[: -len(extension)]
                break

        return value.strip()

    def _normalise_label(self, value):
        if pd.isna(value):
            return None

        if isinstance(value, str):
            cleaned = value.strip().lower()

            text_mapping = {
                "ang": 0,
                "angry": 0,
                "hap": 1,
                "happy": 1,
                "exc": 1,
                "excited": 1,
                "neu": 2,
                "neutral": 2,
                "sad": 3,
            }

            if cleaned in text_mapping:
                return text_mapping[cleaned]

            try:
                numeric_value = int(float(cleaned))
            except ValueError:
                return None
        else:
            numeric_value = int(value)

        if 0 <= numeric_value < 4:
            return numeric_value

        return None

    def _build_face_lookup(self):
        face_lookup = {}

        for emotion_name in self.EMOTION_TO_ID:
            emotion_folder = self.face_dir / emotion_name

            if not emotion_folder.exists():
                print(
                    f"  Warning: missing face emotion folder: "
                    f"{emotion_folder}"
                )
                continue

            for utterance_folder in emotion_folder.iterdir():
                if not utterance_folder.is_dir():
                    continue

                has_valid_frame = any(
                    (utterance_folder / frame_name).is_file()
                    for frame_name in self.FRAME_NAMES
                )

                if has_valid_frame:
                    utterance_id = self._normalise_utterance_id(
                        utterance_folder.name
                    )
                    face_lookup[utterance_id] = utterance_folder

        return face_lookup

    def _build_audio_lookup(self):
        audio_lookup = {}

        for audio_path in self.audio_dir.rglob("*.wav"):
            utterance_id = self._normalise_utterance_id(
                audio_path.name
            )
            audio_lookup.setdefault(
                utterance_id,
                audio_path,
            )

        for audio_path in self.audio_dir.rglob("*.WAV"):
            utterance_id = self._normalise_utterance_id(
                audio_path.name
            )
            audio_lookup.setdefault(
                utterance_id,
                audio_path,
            )

        return audio_lookup

    def _resolve_audio_path(self, row, utterance_id):
        if self.audio_column is not None:
            raw_path = row[self.audio_column]

            if not pd.isna(raw_path):
                candidate = Path(
                    str(raw_path).strip().replace("\\", "/")
                )

                candidates = []

                if candidate.is_absolute():
                    candidates.append(candidate)
                else:
                    candidates.extend(
                        [
                            PROJECT_ROOT / candidate,
                            self.audio_dir / candidate,
                            self.audio_dir / candidate.name,
                        ]
                    )

                for path_candidate in candidates:
                    if path_candidate.is_file():
                        return path_candidate.resolve()

        return self.audio_lookup.get(utterance_id)

    def set_training(self, training):
        self.training = bool(training)

    def __len__(self):
        return len(self.samples)

    def _extract_mfcc(self, audio_path):
        mfcc = extract_mfcc(audio_path)

        normalization_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "speech_normalization.npz"
        )

        if not normalization_path.exists():
            raise FileNotFoundError(
                f"Speech normalization file not found: {normalization_path}"
            )

        stats = np.load(normalization_path)
        mean = stats["mean"]
        std = stats["std"]

        mfcc = (mfcc - mean) / (std + 1e-8)

        return torch.tensor(
            mfcc,
            dtype=torch.float32,
        )

    

    def _load_face_frames(self, face_folder):
        frames = []
        frame_mask = []

        for frame_name in self.FRAME_NAMES:
            image_path = face_folder / frame_name

            if image_path.is_file():
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image_tensor = self.face_transform(image)

                frames.append(image_tensor)
                frame_mask.append(1.0)
            else:
                frames.append(
                    torch.zeros(
                        1,
                        self.face_size,
                        self.face_size,
                        dtype=torch.float32,
                    )
                )
                frame_mask.append(0.0)

        return (
            torch.stack(frames, dim=0),
            torch.tensor(frame_mask, dtype=torch.float32),
        )

    def __getitem__(self, index):
        sample = self.samples[index]

        speech_mfcc = self._extract_mfcc(
            sample["audio_path"]
        )

        face_frames, frame_mask = self._load_face_frames(
            sample["face_folder"]
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return speech_mfcc, face_frames, frame_mask, label



# HELPER FUNCTION: CONFIDENCE-BASED WEIGHTING

def apply_confidence_weights(
    speech_embeddings,
    face_embeddings,
    speech_logits,
    face_logits,
    num_emotions,
    fusion_mode="dynamic",
):
    """
    Apply modality weighting to speech and face embeddings.
    
    fusion_mode:
        "dynamic": entropy-based confidence weighting
        "equal": fixed 0.5/0.5 weighting (baseline comparison)
    """

    if fusion_mode == "equal":
        # ===== EQUAL WEIGHTING BASELINE =====
        # Fixed 0.5/0.5 weights (no entropy calculation)
        weight_speech = torch.full(
            (speech_embeddings.size(0), 1),
            0.5,
            dtype=speech_embeddings.dtype,
            device=device,
        )
        weight_face = torch.full(
            (face_embeddings.size(0), 1),
            0.5,
            dtype=face_embeddings.dtype,
            device=device,
        )
    else:
        # ===== DYNAMIC ENTROPY-BASED WEIGHTING =====
        confidence_speech = torch.as_tensor(
            get_entropy_based_confidence(
                speech_logits,
                num_emotions,
            ),
            dtype=speech_embeddings.dtype,
            device=device,
        )

        confidence_face = torch.as_tensor(
            get_entropy_based_confidence(
                face_logits,
                num_emotions,
            ),
            dtype=face_embeddings.dtype,
            device=device,
        )

        confidence_sum = (
            confidence_speech
            + confidence_face
            + 1e-8
        )

        weight_speech = (
            confidence_speech / confidence_sum
        ).unsqueeze(1)

        weight_face = (
            confidence_face / confidence_sum
        ).unsqueeze(1)

    weighted_speech = (
        weight_speech * speech_embeddings
    )

    weighted_face = (
        weight_face * face_embeddings
    )

    return weighted_speech, weighted_face


print("Confidence weighting function initialized\n")


# INITIALIZE MODELS

print("Initializing models...")

speech_model = SpeechEmotionModel(
    num_emotions=num_emotions
).to(device)

speech_ckpt = (
    PROJECT_ROOT
    / "checkpoints"
    / "speech_iemocap.pth"
)

if speech_ckpt.exists():
    print(
        f"✓ Loading speech model: "
        f"{speech_ckpt.name}"
    )
    speech_model.load_state_dict(
        torch.load(
            speech_ckpt,
            map_location=device,
        )
    )
else:
    print(
        f"⚠ Speech model not found: "
        f"{speech_ckpt}"
    )
    print(
        "  Run: python scripts/train_speech.py"
    )
    raise SystemExit(1)


face_model = FaceEmotionModel(
    num_emotions=num_emotions
).to(device)

face_ckpt = (
    PROJECT_ROOT
    / "checkpoints"
    / "face_iemocap.pth"
)

if face_ckpt.exists():
    print(
        f"✓ Loading face model: "
        f"{face_ckpt.name}"
    )
    face_model.load_state_dict(
        torch.load(
            face_ckpt,
            map_location=device,
        )
    )
else:
    print(
        f"⚠ Face model not found: "
        f"{face_ckpt}"
    )
    print(
        "  Run: python scripts/pretrain_face_fer2013.py"
    )
    print(
        "  Then: python scripts/finetune_face_iemocap.py"
    )
    raise SystemExit(1)


fusion_model = EmbeddingFusion(
    embedding_dim=128,
    num_emotions=num_emotions,
).to(device)

print("✓ Fusion module initialized")


for param in speech_model.parameters():
    param.requires_grad = False

for param in face_model.parameters():
    param.requires_grad = False

print("  Speech model: frozen")
print("  Face model: frozen")
print("  Fusion module: trainable\n")


# LOAD REAL MULTIMODAL DATA

print("Loading real IEMOCAP multimodal data...")
print(
    "  Pairing: speech MFCC + face frames "
    "from same utterance\n"
)

full_dataset = IEMOCAPMultimodalDataset(
    csv_path=(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "metadata"
        / "iemocap_harmonized.csv"
    ),
    audio_dir=(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "IEMOCAP"
        / "IEMOCAP_full_release"
    ),
    face_dir=(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "IEMOCAP_faces"
    ),
    n_mfcc=13,
    target_length=128,
    face_size=48,
    training=False,
)


if len(full_dataset) == 0:
    raise RuntimeError(
        "No paired multimodal samples were found."
    )


# Split into train/val (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(
        random_seed
    ),
)


# =====================================================================
# COMPUTE CLASS WEIGHTS FROM THE PAIRED TRAINING SPLIT
# =====================================================================

print(
    "Computing class weights from paired "
    "IEMOCAP training data..."
)

train_labels = np.asarray(
    [
        full_dataset.labels[index]
        for index in train_dataset.indices
    ],
    dtype=np.int64,
)

class_counts = np.bincount(
    train_labels,
    minlength=num_emotions,
)

if np.any(class_counts == 0):
    raise ValueError(
        "At least one emotion class has no paired training samples. "
        f"Class distribution: {class_counts}"
    )

print(
    f"  Class distribution: "
    f"{class_counts}"
)

total_samples = class_counts.sum()

class_weights = (
    total_samples
    / (num_emotions * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device,
)

print(
    f"  Class weights: "
    f"{class_weights}\n"
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    fusion_model.parameters(),
    lr=learning_rate,
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)


# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

print("✓ Data loaded and split")
print(
    f"  Paired samples: {len(full_dataset)}"
)
print(
    f"  Training samples: {len(train_dataset)}"
)
print(
    f"  Validation samples: {len(val_dataset)}"
)
print(
    f"  Train batches: {len(train_loader)}"
)
print(
    f"  Val batches:   {len(val_loader)}\n"
)


# TRAINING LOOP

speech_model.eval()
face_model.eval()
fusion_model.train()

best_val_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    full_dataset.set_training(True)

    total_loss = 0.0
    num_batches = 0

    # ===== TRAIN =====
    for (
        speech_mfcc,
        face_frames,
        frame_mask,
        labels,
    ) in train_loader:

        speech_mfcc = speech_mfcc.to(device)
        face_frames = face_frames.to(device)
        frame_mask = frame_mask.to(device)
        labels = labels.to(device)

        
        with torch.no_grad():
            speech_embeddings = speech_model(
                speech_mfcc,
                return_embeddings=True,
            )

            speech_logits = speech_model(
                speech_mfcc,
                return_embeddings=False,
            )

            (
                face_embeddings,
                face_logits,
            ) = forward_face_utterance(
                model=face_model,
                frames=face_frames,
                frame_mask=frame_mask,
            )

            (
                weighted_speech,
                weighted_face,
            ) = apply_confidence_weights(
                speech_embeddings,
                face_embeddings,
                speech_logits,
                face_logits,
                num_emotions,
                fusion_mode,
            )

        fusion_output = fusion_model(
            weighted_speech,
            weighted_face,
        )

        logits = fusion_output["logits"]

        loss = criterion(
            logits,
            labels,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            fusion_model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (num_batches % 5) == 0:
            avg_loss = (
                total_loss / num_batches
            )

            print(
                f"  Epoch "
                f"{epoch + 1:2d}/{num_epochs} | "
                f"Batch "
                f"{num_batches:3d}/"
                f"{len(train_loader):3d} | "
                f"Loss: {avg_loss:.4f}"
            )

    if num_batches == 0:
        raise RuntimeError(
            "The training DataLoader returned no batches."
        )

    epoch_loss = (
        total_loss / num_batches
    )

    print(
        f"\n✓ Epoch "
        f"{epoch + 1:2d}/{num_epochs} Complete | "
        f"Avg Loss: {epoch_loss:.4f}"
    )

    # ===== VALIDATION =====
    full_dataset.set_training(False)
    fusion_model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (
            speech_mfcc,
            face_frames,
            frame_mask,
            labels,
        ) in val_loader:
            speech_mfcc = speech_mfcc.to(device)
            face_frames = face_frames.to(device)
            frame_mask = frame_mask.to(device)
            labels = labels.to(device)

            speech_embeddings = speech_model(
                speech_mfcc,
                return_embeddings=True,
            )

            speech_logits = speech_model(
                speech_mfcc,
                return_embeddings=False,
            )

            (
                face_embeddings,
                face_logits,
            ) = forward_face_utterance(
                model=face_model,
                frames=face_frames,
                frame_mask=frame_mask,
            )

            (
                weighted_speech,
                weighted_face,
            ) = apply_confidence_weights(
                speech_embeddings,
                face_embeddings,
                speech_logits,
                face_logits,
                num_emotions,
                fusion_mode,
            )

            fusion_output = fusion_model(
                weighted_speech,
                weighted_face,
            )

            logits = fusion_output["logits"]

            val_loss += criterion(
                logits,
                labels,
            ).item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    if len(val_loader) == 0:
        raise RuntimeError(
            "The validation DataLoader returned no batches."
        )

    val_loss /= len(val_loader)

    val_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"  Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"LR: {current_lr:.6f}\n"
    )

    scheduler.step(
        val_loss
    )

    # ===== EARLY STOPPING =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / f"fusion_model_{fusion_mode}.pth"
        )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            fusion_model.state_dict(),
            checkpoint_path,
        )

        print(
            f"  ✓ Model saved ({fusion_mode}) "
            f"(val loss: {val_loss:.4f})\n"
        )
    else:
        patience_counter += 1

        if patience_counter >= patience:
            print(
                f"  Early stopping at "
                f"epoch {epoch + 1}\n"
            )
            break

    fusion_model.train()


# SAVE FINAL MODEL

print("=" * 70)
print("FUSION TRAINING COMPLETE")
print("=" * 70)

checkpoint_path = (
    PROJECT_ROOT
    / "checkpoints"
    / f"fusion_model_{fusion_mode}.pth"
)

checkpoint_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# Preserve the best validation checkpoint. Only save here if training
# never produced a validation improvement.
if not checkpoint_path.exists():
    torch.save(
        fusion_model.state_dict(),
        checkpoint_path,
    )

print(
    f"\n✓ Best fusion model retained at: "
    f"{checkpoint_path}"
)

print(
    f"✓ Best validation loss: "
    f"{best_val_loss:.4f}"
)

print("✓ Ready for evaluation\n")
