"""
IEMOCAP Multimodal Dataset
Pairs MFCC features with face images from the same utterance
"""

import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.utils.audio_utils import extract_mfcc

# SPECAUGMENT: Data Augmentation for Speech Features

def spec_augment(mfcc, freq_mask_param=30, time_mask_param=40):
    """
    Apply SpecAugment augmentation to MFCC features
    
    SpecAugment applies random masking to make the model robust:
    - Time Masking: Hide random consecutive time frames (silence injection)
    - Frequency Masking: Hide random consecutive frequency bins (filter simulation)
    
    Args:
        mfcc (np.ndarray): MFCC features, shape (1, freq_bins, time_steps)
        freq_mask_param (int): Max number of frequency bins to mask
        time_mask_param (int): Max number of time frames to mask
    
    Returns:
        augmented_mfcc (np.ndarray): Masked MFCC features
    """
    
    mfcc = mfcc.copy()  # Don't modify original
    
    # Get dimensions
    freq_bins = mfcc.shape[1]
    time_steps = mfcc.shape[2]
    
    # Time Masking: randomly mask T consecutive time frames
    t = np.random.randint(0, time_mask_param)
    if t > 0:
        t_start = np.random.randint(0, time_steps - t)
        mfcc[:, :, t_start:t_start + t] = 0
    
    # Frequency Masking: randomly mask F consecutive frequency bins
    f = np.random.randint(0, freq_mask_param)
    if f > 0:
        f_start = np.random.randint(0, freq_bins - f)
        mfcc[:, f_start:f_start + f, :] = 0
    
    return mfcc


class IEMOCAPMultimodalDataset(Dataset):
    """
    Pair one speech utterance with up to three face frames from the
    same IEMOCAP utterance.

    Each sample returns:
        speech_mfcc: [1, 13, target_length]
        face_frames: [3, 1, 48, 48]
        frame_mask: [3]
        label: scalar
    """

    FRAME_NAMES = (
        "start.png",
        "middle.png",
        "end.png",
    )

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
        csv_path="data/processed/metadata/iemocap_harmonized.csv",
        audio_dir="data/raw/IEMOCAP/IEMOCAP_full_release",
        face_dir="data/processed/IEMOCAP_faces",
        n_mfcc=13,
        target_length=384,
        face_size=48,
        training=False,
    ):
        print("Loading real IEMOCAP multimodal data...")
        print("  Pairing: speech MFCC + utterance-level face frames\n")

        self.csv_path = self._resolve_project_path(csv_path)
        self.audio_dir = self._resolve_project_path(audio_dir)
        self.face_dir = self._resolve_project_path(face_dir)

        self.n_mfcc = n_mfcc
        self.target_length = target_length
        self.face_size = face_size
        self.training = training

        # Load speech normalization statistics
        normalization_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "speech_normalization.npz"
        )

        if not normalization_path.exists():
            raise FileNotFoundError(
                f"Speech normalization file not found: {normalization_path}\n"
                "Run python scripts/train_speech.py first to create it."
            )

        normalization_data = np.load(normalization_path)

        self.speech_mean = normalization_data["mean"]
        self.speech_std = normalization_data["std"]

        print(f"  Speech normalization loaded: {normalization_path.name}")
        print(f"  Speech mean shape: {self.speech_mean.shape}")
        print(f"  Speech std shape:  {self.speech_std.shape}")

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
                transforms.Resize(
                    (self.face_size, self.face_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5],
                ),
            ]
        )

        self.face_lookup = self._build_face_lookup()
        self.audio_lookup = self._build_audio_lookup()

        print(f"  Metadata rows: {len(self.metadata)}")
        print(
            f"  Face utterance folders found: "
            f"{len(self.face_lookup)}"
        )
        print(
            f"  Audio files indexed: "
            f"{len(self.audio_lookup)}"
        )
        print(f"  ID column: {self.id_column}")
        print(
            f"  Audio column: "
            f"{self.audio_column or 'not present; using audio index'}"
        )
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

            face_folder = self.face_lookup.get(
                utterance_id
            )

            if face_folder is None:
                unmatched_face_ids.append(
                    utterance_id
                )
                continue

            audio_path = self._resolve_audio_path(
                row=row,
                utterance_id=utterance_id,
            )

            if audio_path is None:
                unmatched_audio_ids.append(
                    utterance_id
                )
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

        print(
            f"\n✓ Loaded {len(self.samples)} "
            f"paired samples from IEMOCAP"
        )

        if self.samples:
            class_counts = np.bincount(
                np.asarray(
                    self.labels,
                    dtype=np.int64,
                ),
                minlength=4,
            )

            print("  Emotion distribution:")

            for emotion_name, label_id in (
                self.EMOTION_TO_ID.items()
            ):
                print(
                    f"    {emotion_name:<8}: "
                    f"{class_counts[label_id]}"
                )
        else:
            print(
                "  Example face IDs:",
                list(self.face_lookup)[:5],
            )

            print(
                "  Example audio IDs:",
                list(self.audio_lookup)[:5],
            )

            print("  Example metadata IDs:")

            for value in self.metadata[
                self.id_column
            ].head(5):
                print(
                    "   ",
                    self._normalise_utterance_id(
                        value
                    ),
                )

            raise RuntimeError(
                "No multimodal samples could be paired.\n"
                "Check the example metadata, face and audio "
                "IDs printed above."
            )

        if unmatched_face_ids:
            print(
                "  Metadata rows without face match: "
                f"{len(set(unmatched_face_ids))}"
            )

        if unmatched_audio_ids:
            print(
                "  Metadata rows without audio match: "
                f"{len(set(unmatched_audio_ids))}"
            )

        print()

    @staticmethod
    def _resolve_project_path(path_value):
        path = Path(path_value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    def _find_column(
        self,
        candidates,
        required,
        purpose,
    ):
        column_lookup = {
            str(column).strip().lower(): column
            for column in self.metadata.columns
        }

        for candidate in candidates:
            match = column_lookup.get(
                candidate.lower()
            )

            if match is not None:
                return match

        if required:
            raise KeyError(
                f"Could not find a column for {purpose}.\n"
                f"Available columns: "
                f"{self.metadata.columns.tolist()}"
            )

        return None

    @staticmethod
    def _normalise_utterance_id(value):
        if pd.isna(value):
            return ""

        value = (
            str(value)
            .strip()
            .replace("\\", "/")
        )

        value = value.split("/")[-1]

        extensions = (
            ".wav",
            ".WAV",
            ".png",
            ".jpg",
            ".jpeg",
            ".avi",
            ".mp4",
        )

        for extension in extensions:
            if value.endswith(extension):
                value = value[
                    : -len(extension)
                ]
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
                numeric_value = int(
                    float(cleaned)
                )
            except ValueError:
                return None
        else:
            try:
                numeric_value = int(value)
            except (TypeError, ValueError):
                return None

        if 0 <= numeric_value < 4:
            return numeric_value

        return None

    def _build_face_lookup(self):
        face_lookup = {}

        for emotion_name in self.EMOTION_TO_ID:
            emotion_folder = (
                self.face_dir / emotion_name
            )

            if not emotion_folder.exists():
                print(
                    "  Warning: missing face "
                    f"emotion folder: {emotion_folder}"
                )
                continue

            for utterance_folder in (
                emotion_folder.iterdir()
            ):
                if not utterance_folder.is_dir():
                    continue

                has_valid_frame = any(
                    (
                        utterance_folder
                        / frame_name
                    ).is_file()
                    for frame_name in self.FRAME_NAMES
                )

                if not has_valid_frame:
                    continue

                utterance_id = (
                    self._normalise_utterance_id(
                        utterance_folder.name
                    )
                )

                face_lookup[
                    utterance_id
                ] = utterance_folder

        return face_lookup

    def _build_audio_lookup(self):
        audio_lookup = {}

        audio_files = list(
            self.audio_dir.rglob("*.wav")
        )

        audio_files.extend(
            self.audio_dir.rglob("*.WAV")
        )

        for audio_path in audio_files:
            utterance_id = (
                self._normalise_utterance_id(
                    audio_path.name
                )
            )

            audio_lookup.setdefault(
                utterance_id,
                audio_path,
            )

        return audio_lookup

    def _resolve_audio_path(
        self,
        row,
        utterance_id,
    ):
        if self.audio_column is not None:
            raw_path = row[self.audio_column]

            if not pd.isna(raw_path):
                candidate = Path(
                    str(raw_path)
                    .strip()
                    .replace("\\", "/")
                )

                candidates = []

                if candidate.is_absolute():
                    candidates.append(candidate)
                else:
                    candidates.extend(
                        [
                            PROJECT_ROOT / candidate,
                            self.audio_dir / candidate,
                            self.audio_dir
                            / candidate.name,
                        ]
                    )

                for path_candidate in candidates:
                    if path_candidate.is_file():
                        return (
                            path_candidate.resolve()
                        )

        return self.audio_lookup.get(
            utterance_id
        )

    def set_training(self, training):
        self.training = bool(training)

    def __len__(self):
        return len(self.samples)

    def _extract_mfcc(self, audio_path):
        """Extract and normalize MFCCs using speech-training statistics."""

        mfcc = extract_mfcc(
            audio_path,
            n_mfcc=self.n_mfcc,
            target_length=self.target_length,
        )

        # extract_mfcc returns [1, 26, 128]
        mfcc_np = mfcc.squeeze(0).cpu().numpy()

        # Saved statistics normally have shape [1, 26, 1]
        mean = np.squeeze(self.speech_mean, axis=0)
        std = np.squeeze(self.speech_std, axis=0)

        mfcc_np = (
            mfcc_np - mean
        ) / std

        return torch.tensor(
            mfcc_np,
            dtype=torch.float32,
        ).unsqueeze(0)

    def _load_face_frames(
        self,
        face_folder,
    ):
        frames = []
        frame_mask = []

        for frame_name in self.FRAME_NAMES:
            image_path = (
                face_folder / frame_name
            )

            if image_path.is_file():
                with Image.open(
                    image_path
                ) as image:
                    image = image.convert("RGB")

                    image_tensor = (
                        self.face_transform(
                            image
                        )
                    )

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
            torch.stack(
                frames,
                dim=0,
            ),
            torch.tensor(
                frame_mask,
                dtype=torch.float32,
            ),
        )

    def __getitem__(self, index):
        sample = self.samples[index]

        speech_mfcc = self._extract_mfcc(
            sample["audio_path"]
        )

        face_frames, frame_mask = (
            self._load_face_frames(
                sample["face_folder"]
            )
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return (
            speech_mfcc,
            face_frames,
            frame_mask,
            label,
        )