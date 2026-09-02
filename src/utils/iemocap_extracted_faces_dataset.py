"""
Dataset for utterance-level IEMOCAP face images.

Expected directory structure:

IEMOCAP_faces/
    angry/
        utterance_id/
            start.png
            middle.png
            end.png
    happy/
        utterance_id/
            start.png
            middle.png
            end.png
    neutral/
        utterance_id/
            start.png
            middle.png
            end.png
    sad/
        utterance_id/
            start.png
            middle.png
            end.png

Each item returns:

frames:
    Shape [3, 1, target_size, target_size]

frame_mask:
    Shape [3]
    1 means the frame exists.
    0 means the frame is missing.

label:
    Scalar emotion label.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IEMOCAPExtractedFacesDataset(Dataset):
    """Load utterance-level face frames extracted from IEMOCAP."""

    LABEL_MAP = {
        "angry": 0,
        "happy": 1,
        "neutral": 2,
        "sad": 3,
    }

    FRAME_NAMES = (
        "start.png",
        "middle.png",
        "end.png",
    )

    def __init__(
        self,
        root_dir,
        split="train",
        target_size=48,
        train_ratio=0.8,
        random_seed=42,
    ):
        """
        Initialise the IEMOCAP face dataset.

        Parameters
        ----------
        root_dir : str or Path
            Root directory containing emotion folders.

        split : str
            Either "train" or "val".

        target_size : int
            Image height and width after resizing.

        train_ratio : float
            Fraction of utterances assigned to training.

        random_seed : int
            Random seed used for reproducible splitting.
        """

        if split not in {"train", "val"}:
            raise ValueError(
                "split must be either 'train' or 'val'."
            )

        if not 0.0 < train_ratio < 1.0:
            raise ValueError(
                "train_ratio must be between 0 and 1."
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.target_size = target_size

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"IEMOCAP face directory does not exist:\n"
                f"{self.root_dir}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=1
                ),
                transforms.Resize(
                    (target_size, target_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5],
                ),
            ]
        )

        all_samples = []

        print(
            f"Scanning face directory:\n"
            f"{self.root_dir}"
        )

        for emotion_name, label_id in self.LABEL_MAP.items():
            emotion_folder = (
                self.root_dir / emotion_name
            )

            if not emotion_folder.exists():
                print(
                    f"Warning: emotion folder missing: "
                    f"{emotion_folder}"
                )
                continue

            emotion_count = 0

            for utterance_folder in sorted(
                emotion_folder.iterdir()
            ):
                if not utterance_folder.is_dir():
                    continue

                available_frames = [
                    utterance_folder / frame_name
                    for frame_name in self.FRAME_NAMES
                    if (
                        utterance_folder / frame_name
                    ).is_file()
                ]

                if not available_frames:
                    continue

                all_samples.append(
                    {
                        "folder": utterance_folder,
                        "label": label_id,
                    }
                )

                emotion_count += 1

            print(
                f"  {emotion_name:<8}: "
                f"{emotion_count} utterances"
            )

        if not all_samples:
            raise FileNotFoundError(
                "No utterance-level face images were found.\n\n"
                f"Directory checked:\n{self.root_dir}\n\n"
                "Expected structure:\n"
                "IEMOCAP_faces/\n"
                "    angry/\n"
                "        utterance_id/\n"
                "            start.png\n"
                "            middle.png\n"
                "            end.png\n"
                "    happy/\n"
                "    neutral/\n"
                "    sad/"
            )

        generator = torch.Generator()
        generator.manual_seed(random_seed)

        random_indices = torch.randperm(
            len(all_samples),
            generator=generator,
        ).tolist()

        split_index = int(
            train_ratio * len(all_samples)
        )

        if split == "train":
            selected_indices = random_indices[
                :split_index
            ]
        else:
            selected_indices = random_indices[
                split_index:
            ]

        self.samples = [
            all_samples[index]
            for index in selected_indices
        ]

        self.labels = [
            sample["label"]
            for sample in self.samples
        ]

        print(
            f"{split.capitalize()} utterances: "
            f"{len(self.samples)}"
        )

    def __len__(self):
        """Return the number of utterances."""

        return len(self.samples)

    def __getitem__(self, index):
        """Return frames, availability mask and label."""

        sample = self.samples[index]

        utterance_folder = sample["folder"]
        label = sample["label"]

        frames = []
        frame_mask = []

        for frame_name in self.FRAME_NAMES:
            image_path = (
                utterance_folder / frame_name
            )

            if image_path.is_file():
                try:
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        image_tensor = self.transform(
                            image
                        )

                except Exception as error:
                    raise RuntimeError(
                        f"Could not load image:\n"
                        f"{image_path}"
                    ) from error

                frames.append(image_tensor)
                frame_mask.append(1.0)

            else:
                empty_image = torch.zeros(
                    1,
                    self.target_size,
                    self.target_size,
                    dtype=torch.float32,
                )

                frames.append(empty_image)
                frame_mask.append(0.0)

        frames_tensor = torch.stack(
            frames,
            dim=0,
        )

        frame_mask_tensor = torch.tensor(
            frame_mask,
            dtype=torch.float32,
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return (
            frames_tensor,
            frame_mask_tensor,
            label_tensor,
        )