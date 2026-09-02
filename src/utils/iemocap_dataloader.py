"""
IEMOCAP DataLoader with MFCC Extraction
Uses processed metadata CSV for cleaner data loading
"""

import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from src.utils.audio_utils import extract_mfcc

# STEP 1: MFCC EXTRACTION
def extract_mfcc(audio_path, n_mfcc=13, sr=16000, n_fft=400, hop_length=160):
    """
    Extract MFCC + Delta MFCC features from audio file
    
    Returns both MFCC and Delta MFCC (first-order derivative):
    - MFCC: Static spectral features (what the audio "looks like")
    - Delta MFCC: Rate of change features (how the audio evolves over time)
    Together they capture both static and dynamic aspects of emotion
    
    Args:
        audio_path (str): Path to audio file (.wav)
        n_mfcc (int): Number of MFCC coefficients (default: 13)
        sr (int): Sampling rate (default: 16000 Hz)
        n_fft (int): FFT window size (default: 400)
        hop_length (int): Number of samples between frames (default: 160)
    
    Returns:
        mfcc_combined (np.ndarray): MFCC + Delta MFCC, shape (2*n_mfcc, time_steps)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Extract Delta MFCC (first-order derivative)
        delta_mfcc = librosa.feature.delta(mfcc)
        
        # Concatenate MFCC + Delta MFCC
        mfcc_combined = np.vstack([mfcc, delta_mfcc])
        
        return mfcc_combined
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def pad_mfcc(mfcc, target_length=128):
    """
    Pad or truncate MFCC to fixed length
    
    Why? Neural networks need fixed input sizes
    - Short utterances: pad with zeros
    - Long utterances: truncate to target length
    
    Args:
        mfcc (np.ndarray): MFCC features, shape (n_mfcc, time_steps)
        target_length (int): Target time steps (default: 128)
    
    Returns:
        padded_mfcc (np.ndarray): Padded MFCC, shape (n_mfcc, target_length)
    """
    if mfcc is None:
        return None
    
    current_length = mfcc.shape[1]
    
    if current_length >= target_length:
        # Truncate if too long
        return mfcc[:, :target_length]
    else:
        # Pad with zeros if too short
        pad_width = ((0, 0), (0, target_length - current_length))
        return np.pad(mfcc, pad_width, mode='constant', constant_values=0)


# STEP 2: EMOTION LABELS
# Emotion mapping - from processed CSV
EMOTION_NAMES = {
    0: 'angry',
    1: 'happy', 
    2: 'neutral',
    3: 'sad'
}

EMOTION_ID_MAP = {v: k for k, v in EMOTION_NAMES.items()}  # Reverse mapping


# STEP 3: DATASET CLASS
class IEMOCAPDataset(Dataset):
    """
    PyTorch Dataset for IEMOCAP audio data using processed metadata
    
    Uses CSV metadata for efficient loading:
    - Reads utterance IDs and labels from CSV
    - Finds audio files using utterance IDs
    - Extracts MFCC features
    - Normalizes features
    """
    
    def __init__(self, data_dir, csv_path, n_mfcc=13, target_length=128, normalize=True):
        """
        Initialize IEMOCAP Dataset from processed metadata CSV
        
        Args:
            data_dir (str): Path to IEMOCAP_full_release directory
            csv_path (str): Path to iemocap_harmonized.csv file
            n_mfcc (int): Number of MFCC coefficients to extract (actual output = 2*n_mfcc due to delta)
            target_length (int): Fixed MFCC time length
            normalize (bool): Whether to normalize features
        
        Note: With delta features enabled, output shape = (2*n_mfcc, target_length)
              Example: n_mfcc=13 → output (26, 128)
        """
        self.data_dir = data_dir
        self.n_mfcc = n_mfcc
        self.target_length = target_length
        self.normalize = normalize
        
        # Load metadata from CSV
        print(f"Loading metadata from: {csv_path}")
        self.metadata_df = pd.read_csv(csv_path)
        print(f"Found {len(self.metadata_df)} samples in CSV\n")
        
        # Find audio files based on utterance IDs
        self.audio_files = self._find_audio_files()
        print(f"Located {len(self.audio_files)} audio files\n")
        
        # Load and cache all MFCC features and labels
        self.mfcc_features = []
        self.labels = []
        self._load_all_features()
    
    def _find_audio_files(self):
        """
        Find audio file paths using utterance IDs from CSV
        
        Maps utterance_id (e.g., 'Ses01F_impro01_F000') to actual .wav file path
        """
        audio_files = []
        
        for idx, row in self.metadata_df.iterrows():
            utterance_id = row['utterance_id']
            
            # Search pattern: find SessionX folder and match utterance
            # Example: Ses01F_impro01_F000 -> Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
            pattern = os.path.join(
                self.data_dir, 
                'Session*', 
                'sentences', 
                'wav', 
                '*', 
                f'{utterance_id}.wav'
            )
            
            files = glob.glob(pattern)
            
            if files:
                audio_files.append((files[0], row['label_id']))
            else:
                # Try alternative naming pattern
                pattern2 = os.path.join(
                    self.data_dir,
                    'Session*',
                    'sentences',
                    'wav',
                    f'{utterance_id[:16]}*',  # e.g., Ses01F_impro01
                    f'{utterance_id}.wav'
                )
                files2 = glob.glob(pattern2)
                if files2:
                    audio_files.append((files2[0], row['label_id']))
                else:
                    print(f"Warning: Could not find audio file for {utterance_id}")
        
        return audio_files
    
    def _load_all_features(self):
        """
        Load all MFCC features and labels from audio files
        Caches features for faster training
        """
        print("Extracting MFCC features (this may take a few minutes)...")
        
        failed_count = 0
        
        for idx, (audio_file, label_id) in enumerate(self.audio_files):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(self.audio_files)} files")
            
            # Extract MFCC
            mfcc = extract_mfcc(audio_file, n_mfcc=self.n_mfcc)
            
            if mfcc is None:
                failed_count += 1
                continue
            
            # Pad to fixed length
            mfcc = pad_mfcc(mfcc, target_length=self.target_length)
            
            # Store
            self.mfcc_features.append(mfcc)
            self.labels.append(label_id)
        
        self.mfcc_features = np.array(self.mfcc_features)
        self.labels = np.array(self.labels)
        
        # Normalize features if requested
        if self.normalize:
            self._normalize_features()
        
        print(f"Loaded {len(self.mfcc_features)} features successfully!")
        if failed_count > 0:
            print(f"Failed to process {failed_count} files\n")
        else:
            print()
    
    def _normalize_features(self):
        """Normalize features using mean and std"""
        mean = self.mfcc_features.mean(axis=(0, 2), keepdims=True)
        std = self.mfcc_features.std(axis=(0, 2), keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        self.mfcc_features = (self.mfcc_features - mean) / std
    def _normalize_features(self):
        """Normalize features using dataset-level mean and standard deviation."""

        self.feature_mean = self.mfcc_features.mean(
            axis=(0, 2),
            keepdims=True,
        )

        self.feature_std = self.mfcc_features.std(
            axis=(0, 2),
            keepdims=True,
        )

        self.feature_std = np.where(
            self.feature_std == 0,
            1,
            self.feature_std,
        )

        self.mfcc_features = (
            self.mfcc_features - self.feature_mean
        ) / self.feature_std
    def __len__(self):
        """Return total number of samples"""
        return len(self.mfcc_features)
    
    def __getitem__(self, idx):
        """
        Return one sample
        
        Args:
            idx (int): Sample index
        
        Returns:
            features (torch.Tensor): MFCC features, shape (1, n_mfcc, target_length)
            label (torch.Tensor): Emotion label (0-3)
        """
        # Get MFCC features
        mfcc = self.mfcc_features[idx]
        
        # Convert to tensor and add channel dimension
        # Shape: (1, n_mfcc, target_length) - for CNN input
        features = torch.FloatTensor(mfcc).unsqueeze(0)
        
        # Get label
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return features, label


# STEP 4: CREATE DATALOADER
def create_dataloaders(data_dir, csv_path, batch_size=32, test_split=0.2, val_split=0.1, 
                       num_workers=0, n_mfcc=13, target_length=128):
    """
    Create train, validation, and test DataLoaders for IEMOCAP
    
    Args:
        data_dir (str): Path to IEMOCAP_full_release directory (raw audio files)
        csv_path (str): Path to iemocap_harmonized.csv (processed metadata)
        batch_size (int): Batch size for training
        test_split (float): Fraction of data for testing (0.0-1.0)
        val_split (float): Fraction of remaining data for validation (0.0-1.0)
        num_workers (int): Number of workers for data loading (0 = main process)
        n_mfcc (int): Number of MFCC coefficients
        target_length (int): Fixed MFCC time length
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
    """
    
    # Create dataset from CSV metadata
    dataset = IEMOCAPDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        n_mfcc=n_mfcc,
        target_length=target_length,
        normalize=True
    )
    # Save normalization statistics for multimodal evaluation
    normalization_path = Path("checkpoints") / "speech_normalization.npz"

    normalization_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        normalization_path,
        mean=dataset.feature_mean,
        std=dataset.feature_std,
    )

    print(f"Speech normalization saved to: {normalization_path}")

    # Get total samples
    total_samples = len(dataset)
    
    # Calculate split sizes
    # Calculate split sizes
    test_size = int(test_split * total_samples)

    remaining_size = total_samples - test_size
    val_size = int(val_split * remaining_size)

    # Assign every remaining sample to training
    train_size = total_samples - val_size - test_size

    # Confirm that no samples are missing
    assert train_size + val_size + test_size == total_samples, (
        f"Split error: train={train_size}, val={val_size}, "
        f"test={test_size}, total={total_samples}"
    )

    print("\nDataset split:")
    print(f"  Total samples:      {total_samples}")
    print(f"  Training samples:   {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Test samples:       {test_size}")
    print(
        f"  Split total:        "
        f"{train_size + val_size + test_size}"
    )

    # Make the split reproducible
    split_generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=split_generator
    )
    
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Print dataset statistics
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples:      {total_samples}")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"\nBatch size:         {batch_size}")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")
    
    # Print emotion distribution
    print(f"\nEmotion distribution:")
    unique, counts = np.unique(dataset.labels, return_counts=True)
    for emotion_id, count in zip(unique, counts):
        emotion_name = EMOTION_NAMES.get(emotion_id, 'unknown')
        percentage = (count / total_samples) * 100
        print(f"  {emotion_name:8s}: {count:5d} ({percentage:5.1f}%)")
    
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader


# STEP 5: FER2013 FACE DATASET
class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for FER2013 face images
    
    FER2013: Facial Expression Recognition 2013 dataset
    - 28,709 training images
    - 3,589 test images
    - 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)
    
    NOTE: We map 7 classes to our 4 classes:
    - angry → 0
    - happy → 1
    - neutral → 2
    - sad → 3
    """
    
    # Map 7-class emotions to 4-class emotions
    EMOTION_7_TO_4 = {
        0: 0,  # angry → angry
        1: None,  # disgust (skip)
        2: None,  # fear (skip)
        3: 1,  # happy → happy
        4: 2,  # neutral → neutral
        5: 3,  # sad → sad
        6: None  # surprise (skip)
    }
    
    EMOTION_NAMES_7 = {
        0: 'angry', 1: 'disgust', 2: 'fear',
        3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
    }
    
    # Map emotion names to 4-class IDs
    EMOTION_NAME_TO_4 = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3
    }
    
    def __init__(self, data_dir, split='train', target_size=48):
        """
        Initialize FER2013 Dataset
        
        Args:
            data_dir (str): Path to FER2013 folder (contains train/ and test/ subfolders)
            split (str): 'train' or 'test'
            target_size (int): Target image size (default: 48x48)
        """
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        self.images = []
        self.labels = []
        
        # Load images from directory structure
        self._load_images()
    
    def _load_images(self):
        """Load images from split directory with emotion names as folder names"""
        split_path = os.path.join(self.data_dir, self.split)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"FER2013 {self.split} folder not found at: {split_path}")
        
        print(f"Loading FER2013 {self.split} data from: {split_path}")
        
        # FER2013 directory structure: emotion_name_folders/images
        emotion_folders = sorted([d for d in os.listdir(split_path) 
                                 if os.path.isdir(os.path.join(split_path, d))])
        
        for emotion_folder in emotion_folders:
            # Try parsing as emotion name first (angry, happy, etc.)
            emotion_name = emotion_folder.lower()
            
            # Skip emotions not in our 4-class mapping
            if emotion_name not in self.EMOTION_NAME_TO_4:
                print(f"  Skipping emotion folder: {emotion_folder}")
                continue
            
            emotion_id_4 = self.EMOTION_NAME_TO_4[emotion_name]
            emotion_path = os.path.join(split_path, emotion_folder)
            
            # Look for both .png and .jpg files
            image_files = sorted(glob.glob(os.path.join(emotion_path, '*.png')))
            image_files += sorted(glob.glob(os.path.join(emotion_path, '*.jpg')))
            image_files += sorted(glob.glob(os.path.join(emotion_path, '*.jpeg')))
            
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(emotion_id_4)
            
            if image_files:
                print(f"  Loaded {len(image_files)} {emotion_name} images")
        
        print(f"Total images loaded: {len(self.images)}\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return one sample
        
        Args:
            idx (int): Sample index
        
        Returns:
            image (torch.Tensor): Grayscale image, shape (1, 48, 48)
            label (torch.Tensor): Emotion label (0-3)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load image (grayscale)
            from PIL import Image
            img = Image.open(img_path).convert('L')  # L = grayscale
            
            # Resize to target size
            img = img.resize((self.target_size, self.target_size))
            
            # Convert to tensor and normalize
            img_tensor = torch.FloatTensor(np.array(img)).unsqueeze(0) / 255.0
            
            return img_tensor, torch.LongTensor([label])[0]
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(1, self.target_size, self.target_size), torch.LongTensor([label])[0]


# STEP 6: UNIFIED DATALOADER FUNCTION
def get_dataloaders(batch_size=32, num_workers=0, test_split=0.2):
    """
    Create real data loaders for all modalities
    
    Returns:
        speech_loader: DataLoader for IEMOCAP speech data (training)
        face_loader: DataLoader for FER2013 face data (training)
        multimodal_loader: Combined loader (test/evaluation)
    
    NOTE: Uses real IEMOCAP and FER2013 datasets, not dummy data
    """
    
    # Paths to real data
    iemocap_raw_path = r"data/raw/IEMOCAP/IEMOCAP_full_release"
    iemocap_csv_path = r"data/processed/metadata/iemocap_harmonized.csv"
    fer2013_path = r"data/raw/FER2013"
    
    print("="*70)
    print("LOADING REAL DATASETS")
    print("="*70)
    
    # ===== SPEECH DATA (IEMOCAP) =====
    print("\n1. IEMOCAP SPEECH DATA")
    print("-"*70)
    train_speech, val_speech, test_speech = create_dataloaders(
        data_dir=iemocap_raw_path,
        csv_path=iemocap_csv_path,
        batch_size=batch_size,
        test_split=test_split,
        val_split=0.1,
        num_workers=num_workers,
        n_mfcc=13,
        target_length=128
    )
    
    # ===== FACE DATA (FER2013) =====
    print("\n2. FER2013 FACE DATA")
    print("-"*70)
    fer2013_train = FER2013Dataset(fer2013_path, split='train', target_size=48)
    fer2013_test = FER2013Dataset(fer2013_path, split='test', target_size=48)
    
    # Split training data for train/val
    train_size = int(0.9 * len(fer2013_train))
    val_size = len(fer2013_train) - train_size
    fer2013_train_split, fer2013_val_split = torch.utils.data.random_split(
        fer2013_train, [train_size, val_size]
    )
    
    train_face = DataLoader(fer2013_train_split, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_face = DataLoader(fer2013_val_split, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_face = DataLoader(fer2013_test, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    print("\n3. DATASET SUMMARY")
    print("-"*70)
    print(f"Speech training batches:   {len(train_speech)}")
    print(f"Face training batches:     {len(train_face)}")
    print(f"Test batches:              {len(test_speech)}")
    print("="*70 + "\n")
    
    # Return loaders
    # For training: use train loaders
    # For evaluation: use test loaders
    return {
        'train_speech': train_speech,
        'train_face': train_face,
        'val_speech': val_speech,
        'val_face': val_face,
        'test_speech': test_speech,
        'test_face': test_face
    }


# EXAMPLE USAGE
if __name__ == "__main__":
    # Paths
    iemocap_raw_path = r"c:\Users\etimo\Desktop\Kemocity\LTU\Final Project\Project Plan\Multimodal_Emotion_Recognition\data\raw\IEMOCAP\IEMOCAP_full_release"
    csv_path = r"c:\Users\etimo\Desktop\Kemocity\LTU\Final Project\Project Plan\Multimodal_Emotion_Recognition\data\processed\metadata\iemocap_harmonized.csv"
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=iemocap_raw_path,          # Raw audio files location
        csv_path=csv_path,                  # Processed metadata CSV
        batch_size=32,
        test_split=0.2,      # 20% for testing
        val_split=0.1,       # 10% of training for validation
        num_workers=0,       # Number of CPU workers (0 = main process)
        n_mfcc=13,          # 13 MFCC coefficients
        target_length=128   # Fixed time length
    )
    
    # Test: Get a batch from training data
    print("Sample batch from training data:")
    print("-" * 60)
    features, labels = next(iter(train_loader))
    
    print(f"Batch size:           {features.shape[0]}")
    print(f"Feature shape:        {features.shape}")
    print(f"  - Channels:         {features.shape[1]} (MFCC)")
    print(f"  - MFCC coefficients: {features.shape[2]}")
    print(f"  - Time steps:       {features.shape[3]}")
    print(f"\nLabels shape:         {labels.shape}")
    print(f"Label values:         {labels.tolist()}")
    print(f"Label names:          {[EMOTION_NAMES[l.item()] for l in labels]}")
    print("-" * 60)
