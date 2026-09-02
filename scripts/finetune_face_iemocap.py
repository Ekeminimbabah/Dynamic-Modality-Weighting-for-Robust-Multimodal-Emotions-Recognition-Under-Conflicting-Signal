
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# PROJECT IMPORTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.face_model import FaceEmotionModel
from src.utils.iemocap_extracted_faces_dataset import (
    IEMOCAPExtractedFacesDataset,
)


# CONFIGURATION
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_EPOCHS = 40
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_EMOTIONS = 4
EARLY_STOPPING_PATIENCE = 5
MIN_ACCURACY_IMPROVEMENT = 0.001
RANDOM_SEED = 42

PRETRAIN_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "face_fer2013.pth"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "face_iemocap.pth"
)

IEMOCAP_FACES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IEMOCAP_faces"
)


# REPRODUCIBILITY
def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# MODEL HELPER FUNCTIONS
def get_classifier(model: nn.Module) -> nn.Module:
    """
    Return the classification layer used after the face embedding.

    The function supports common classifier names. It raises a clear
    error if none of them exists.
    """

    possible_names = [
        "classifier",
        "fc",
        "output_layer",
        "emotion_classifier",
    ]

    for name in possible_names:
        classifier = getattr(model, name, None)

        if classifier is not None:
            return classifier

    raise AttributeError(
        "A classification layer could not be found in FaceEmotionModel.\n"
        "Expected one of these attributes:\n"
        "  classifier\n"
        "  fc\n"
        "  output_layer\n"
        "  emotion_classifier\n\n"
        "Open src/models/face_model.py and check the name of the final "
        "classification layer."
    )


def extract_frame_embeddings(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Extract embeddings from a batch of individual face images.

    Parameters
    ----------
    model:
        Face emotion model.

    images:
        Tensor with shape:
        [number_of_images, channels, height, width]

    Returns
    -------
    torch.Tensor
        Frame embeddings with shape:
        [number_of_images, embedding_dimension]
    """

    model_output = model(
        images,
        return_embeddings=True,
    )

    # Some models return embeddings directly.
    if isinstance(model_output, torch.Tensor):
        embeddings = model_output

    # Some models return something such as:
    # (logits, embeddings)
    elif isinstance(model_output, (tuple, list)):
        if len(model_output) < 2:
            raise ValueError(
                "FaceEmotionModel returned a tuple or list, but it "
                "did not contain both logits and embeddings."
            )

        embeddings = model_output[-1]

    # Some models return a dictionary.
    elif isinstance(model_output, dict):
        if "embeddings" in model_output:
            embeddings = model_output["embeddings"]
        elif "embedding" in model_output:
            embeddings = model_output["embedding"]
        else:
            raise KeyError(
                "FaceEmotionModel returned a dictionary, but it did "
                "not contain 'embedding' or 'embeddings'."
            )

    else:
        raise TypeError(
            "Unsupported output from FaceEmotionModel when "
            "return_embeddings=True."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected face embeddings with two dimensions:\n"
            "[number_of_images, embedding_dimension]\n"
            f"Received shape: {tuple(embeddings.shape)}"
        )

    return embeddings


def forward_utterance(
    model: nn.Module,
    frames: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process up to three face frames belonging to each utterance.

    Parameters
    ----------
    model:
        Face emotion model.

    frames:
        Face images with shape:
        [batch, number_of_frames, channels, height, width]

    frame_mask:
        Frame availability mask with shape:
        [batch, number_of_frames]

        A value of 1 means that the frame exists.
        A value of 0 means that the frame is missing.

    Returns
    -------
    logits:
        Utterance emotion predictions with shape:
        [batch, number_of_emotions]

    utterance_embeddings:
        Averaged face embeddings with shape:
        [batch, embedding_dimension]
    """

    if frames.ndim != 5:
        raise ValueError(
            "Expected frames with shape "
            "[batch, frames, channels, height, width].\n"
            f"Received shape: {tuple(frames.shape)}"
        )

    if frame_mask.ndim != 2:
        raise ValueError(
            "Expected frame_mask with shape [batch, frames].\n"
            f"Received shape: {tuple(frame_mask.shape)}"
        )

    (
        batch_size,
        number_of_frames,
        channels,
        height,
        width,
    ) = frames.shape

    expected_mask_shape = (
        batch_size,
        number_of_frames,
    )

    if tuple(frame_mask.shape) != expected_mask_shape:
        raise ValueError(
            "The frame mask does not match the frame tensor.\n"
            f"Expected mask shape: {expected_mask_shape}\n"
            f"Received mask shape: {tuple(frame_mask.shape)}"
        )

    # Convert:
    # [batch, frames, channels, height, width]
    # into:
    # [batch * frames, channels, height, width]
    flat_frames = frames.reshape(
        batch_size * number_of_frames,
        channels,
        height,
        width,
    )

    frame_embeddings = extract_frame_embeddings(
        model=model,
        images=flat_frames,
    )

    embedding_dimension = frame_embeddings.size(1)

    # Convert back to:
    # [batch, frames, embedding_dimension]
    frame_embeddings = frame_embeddings.reshape(
        batch_size,
        number_of_frames,
        embedding_dimension,
    )

    # Change the mask from:
    # [batch, frames]
    # to:
    # [batch, frames, 1]
    expanded_mask = frame_mask.unsqueeze(-1).to(
        dtype=frame_embeddings.dtype
    )

    # Ignore the embeddings belonging to missing frames.
    masked_embeddings = (
        frame_embeddings * expanded_mask
    )

    summed_embeddings = masked_embeddings.sum(
        dim=1
    )

    valid_frame_counts = expanded_mask.sum(
        dim=1
    ).clamp(min=1.0)

    utterance_embeddings = (
        summed_embeddings / valid_frame_counts
    )

    classifier = get_classifier(model)

    logits = classifier(
        utterance_embeddings
    )

    if logits.ndim != 2:
        raise ValueError(
            "The classifier output should have shape "
            "[batch, number_of_emotions].\n"
            f"Received shape: {tuple(logits.shape)}"
        )

    return logits, utterance_embeddings


# CLASS WEIGHTS
def calculate_class_weights(
    labels: list[int],
    number_of_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Calculate inverse-frequency class weights from the training split.
    """

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    class_counts = np.bincount(
        labels_array,
        minlength=number_of_classes,
    )

    missing_classes = np.where(
        class_counts == 0
    )[0]

    if len(missing_classes) > 0:
        raise ValueError(
            "The training split contains no samples for these "
            f"class IDs: {missing_classes.tolist()}"
        )

    total_samples = class_counts.sum()

    class_weights = total_samples / (
        number_of_classes * class_counts
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    return class_weights_tensor, class_counts


# TRAINING
def main() -> None:
    """Run IEMOCAP face-model fine-tuning."""

    set_random_seed(
        RANDOM_SEED
    )

    print("=" * 70)
    print("STAGE 2: FINE-TUNE FACE MODEL ON IEMOCAP")
    print("=" * 70)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"Learning rate: {LEARNING_RATE} "
        "(transfer learning)\n"
    )

    # CHECK REQUIRED PATHS
    if not PRETRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained model not found:\n{PRETRAIN_PATH}\n\n"
            "Run this first:\n"
            "python scripts/pretrain_face_fer2013.py"
        )

    if not IEMOCAP_FACES_PATH.exists():
        raise FileNotFoundError(
            f"IEMOCAP face directory not found:\n"
            f"{IEMOCAP_FACES_PATH}"
        )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # INITIALISE MODEL
    face_model = FaceEmotionModel(
        num_emotions=NUM_EMOTIONS
    ).to(DEVICE)

    print(
        f"Loading pretrained weights from: "
        f"{PRETRAIN_PATH.name}"
    )

    pretrained_state = torch.load(
        PRETRAIN_PATH,
        map_location=DEVICE,
    )

    face_model.load_state_dict(
        pretrained_state
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in face_model.parameters()
    )

    print(
        f"Model parameters: "
        f"{number_of_parameters:,}\n"
    )

    optimizer = optim.Adam(
        face_model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # LOAD DATA

    print(
        "Loading utterance-level IEMOCAP face data..."
    )

    iemocap_train = IEMOCAPExtractedFacesDataset(
        root_dir=IEMOCAP_FACES_PATH,
        split="train",
        target_size=48,
        train_ratio=0.8,
        random_seed=RANDOM_SEED,
    )

    iemocap_val = IEMOCAPExtractedFacesDataset(
        root_dir=IEMOCAP_FACES_PATH,
        split="val",
        target_size=48,
        train_ratio=0.8,
        random_seed=RANDOM_SEED,
    )
    train_face = DataLoader(
        dataset=iemocap_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_face = DataLoader(
        dataset=iemocap_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_face) == 0:
        raise RuntimeError(
            "The training DataLoader contains no batches."
        )

    if len(val_face) == 0:
        raise RuntimeError(
            "The validation DataLoader contains no batches."
        )

    print("IEMOCAP face data loaded")
    print(
        f"Training utterances: "
        f"{len(iemocap_train)}"
    )
    print(
        f"Validation utterances: "
        f"{len(iemocap_val)}"
    )
    print(
        f"Training batches: "
        f"{len(train_face)}"
    )
    print(
        f"Validation batches: "
        f"{len(val_face)}\n"
    )

    # VERIFY THE BATCH SHAPES
    (
        sample_frames,
        sample_frame_mask,
        sample_labels,
    ) = next(iter(train_face))

    print("Batch shape check:")
    print(
        f"  Frames:     "
        f"{tuple(sample_frames.shape)}"
    )
    print(
        f"  Frame mask: "
        f"{tuple(sample_frame_mask.shape)}"
    )
    print(
        f"  Labels:     "
        f"{tuple(sample_labels.shape)}\n"
    )

    expected_frame_dimensions = 5

    if sample_frames.ndim != expected_frame_dimensions:
        raise ValueError(
            "The dataset is not returning utterance-level frames.\n"
            "Expected shape similar to:\n"
            "[32, 3, 1, 48, 48]\n"
            f"Received: {tuple(sample_frames.shape)}"
        )
    
    # CLASS WEIGHTS
    print(
        "Computing class weights from the actual "
        "IEMOCAP face-training split..."
    )

    (
        class_weights,
        class_counts,
    ) = calculate_class_weights(
        labels=iemocap_train.labels,
        number_of_classes=NUM_EMOTIONS,
        device=DEVICE,
    )

    emotion_names = [
        "angry",
        "happy",
        "neutral",
        "sad",
    ]

    for emotion_name, count in zip(
        emotion_names,
        class_counts,
    ):
        print(
            f"  {emotion_name:<8}: "
            f"{count}"
        )

    print(
        f"\nClass weights: "
        f"{class_weights.detach().cpu()}\n"
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # TEST ONE FORWARD PASS BEFORE TRAINING
    print(
        "Testing one utterance-level forward pass..."
    )

    face_model.eval()

    with torch.no_grad():
        test_frames = sample_frames.to(
            DEVICE
        )

        test_mask = sample_frame_mask.to(
            DEVICE
        )

        (
            test_logits,
            test_embeddings,
        ) = forward_utterance(
            model=face_model,
            frames=test_frames,
            frame_mask=test_mask,
        )

    print(
        f"  Utterance embeddings: "
        f"{tuple(test_embeddings.shape)}"
    )
    print(
        f"  Emotion logits:       "
        f"{tuple(test_logits.shape)}"
    )

    if test_logits.size(1) != NUM_EMOTIONS:
        raise ValueError(
            "The face classifier does not produce four emotion "
            f"classes. Received {test_logits.size(1)} outputs."
        )

    print(
        "Forward-pass test completed successfully.\n"
    )

    # FINE-TUNING
    print("=" * 70)
    print("FINE-TUNING")
    print("=" * 70)
    print()

    best_val_loss = float("inf")
    best_val_accuracy = -1.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):

        # TRAINING
        face_model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(
            train_face
        ):
            (
                frames,
                frame_mask,
                labels,
            ) = batch

            frames = frames.to(
                DEVICE,
                non_blocking=True,
            )

            frame_mask = frame_mask.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                outputs,
                _,
            ) = forward_utterance(
                model=face_model,
                frames=frames,
                frame_mask=frame_mask,
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                face_model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

            if (
                batch_idx == 0
                or (batch_idx + 1) % 10 == 0
                or (batch_idx + 1) == len(train_face)
            ):
                running_loss = (
                    train_loss
                    / (batch_idx + 1)
                )

                running_accuracy = (
                    100.0
                    * train_correct
                    / max(train_total, 1)
                )

                print(
                    f"  Epoch "
                    f"{epoch + 1:2d}/{NUM_EPOCHS} | "
                    f"Batch "
                    f"{batch_idx + 1:4d}/"
                    f"{len(train_face):4d} | "
                    f"Loss: {running_loss:.4f} | "
                    f"Accuracy: "
                    f"{running_accuracy:.2f}%"
                )

        epoch_train_loss = (
            train_loss / len(train_face)
        )

        epoch_train_accuracy = (
            100.0
            * train_correct
            / max(train_total, 1)
        )

        # VALIDATION
        face_model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_face:
                (
                    frames_val,
                    frame_mask_val,
                    labels_val,
                ) = batch

                frames_val = frames_val.to(
                    DEVICE,
                    non_blocking=True,
                )

                frame_mask_val = (
                    frame_mask_val.to(
                        DEVICE,
                        non_blocking=True,
                    )
                )

                labels_val = labels_val.to(
                    DEVICE,
                    non_blocking=True,
                )

                (
                    val_outputs,
                    _,
                ) = forward_utterance(
                    model=face_model,
                    frames=frames_val,
                    frame_mask=frame_mask_val,
                )

                loss = criterion(
                    val_outputs,
                    labels_val,
                )

                val_loss += loss.item()

                val_predictions = (
                    val_outputs.argmax(dim=1)
                )

                val_correct += (
                    val_predictions
                    == labels_val
                ).sum().item()

                val_total += labels_val.size(0)

        epoch_val_loss = (
            val_loss / len(val_face)
        )

        epoch_val_accuracy = (
            100.0
            * val_correct
            / max(val_total, 1)
        )

        scheduler.step(
            epoch_val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch "
            f"{epoch + 1:2d}/{NUM_EPOCHS} complete\n"
            f"  Train loss:     "
            f"{epoch_train_loss:.4f}\n"
            f"  Train accuracy: "
            f"{epoch_train_accuracy:.2f}%\n"
            f"  Val loss:       "
            f"{epoch_val_loss:.4f}\n"
            f"  Val accuracy:   "
            f"{epoch_val_accuracy:.2f}%\n"
            f"  Learning rate:  "
            f"{current_lr:.8f}\n"
        )

        # CHECKPOINT AND EARLY STOPPING
        accuracy_improved = (
            epoch_val_accuracy
            > best_val_accuracy
            + MIN_ACCURACY_IMPROVEMENT
        )

        if accuracy_improved:
            best_val_accuracy = (
                epoch_val_accuracy
            )

            best_val_loss = (
                epoch_val_loss
            )

            patience_counter = 0

            torch.save(
                face_model.state_dict(),
                CHECKPOINT_PATH,
            )

            print(
                "  Best model saved\n"
                f"  Validation accuracy: "
                f"{best_val_accuracy:.2f}%\n"
                f"  Validation loss: "
                f"{best_val_loss:.4f}\n"
            )

        else:
            patience_counter += 1

            print(
                "  No validation-accuracy "
                "improvement "
                f"({patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE})\n"
            )

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    "  Early stopping activated "
                    f"at epoch {epoch + 1}.\n"
                )
                break

    #  FINE-TUNING COMPLETE
    print("=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest model retained at:\n"
        f"{CHECKPOINT_PATH}"
    )
    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )
    print(
        "The model is ready for "
        "multimodal fusion training.\n"
    )


if __name__ == "__main__":
    main()