import librosa
import numpy as np
import torch


def extract_mfcc(
    audio_path,
    sample_rate=16000,
    n_mfcc=13,
    target_length=128,
):
    """
    Return MFCC + delta-MFCC features with shape [1, 26, 128].
    
    MFCC captures spectral characteristics of speech.
    Delta MFCC captures rate of change, representing temporal dynamics.
    Together they form a 26-dimensional feature vector at each time frame.
    
    Parameters:
        audio_path (str): Path to WAV audio file
        sample_rate (int): Resample audio to this rate (16kHz standard for speech)
        n_mfcc (int): Number of MFCC coefficients to extract (13 is standard)
        target_length (int): Fixed time dimension for batching (128 frames)
    
    Returns:
        torch.Tensor: Shape [1, 26, 128] - MFCC + Delta MFCC for batch processing
    """
    # Load audio file and convert to mono at specified sample rate
    audio, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
    )

    # Extract MFCC features: converts waveform to frequency representation
    # n_fft=400 (25ms window), hop_length=160 (10ms hop) standard for 16kHz audio
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=400,
        hop_length=160,
    )

    # Compute first-order delta (derivative) for temporal dynamics
    # Captures how MFCC coefficients change over time - critical for emotion cues
    delta_mfcc = librosa.feature.delta(mfcc)

    # Concatenate static and delta features: (13, time) + (13, time) -> (26, time)
    combined_mfcc = np.vstack((mfcc, delta_mfcc))

    # Enforce fixed length for batch processing: pad or truncate to target_length
    if combined_mfcc.shape[1] < target_length:
        # Pad short utterances with zeros at end
        padding = target_length - combined_mfcc.shape[1]

        combined_mfcc = np.pad(
            combined_mfcc,
            pad_width=((0, 0), (0, padding)),
            mode="constant",
        )
    else:
        # Truncate long utterances to fixed size
        combined_mfcc = combined_mfcc[:, :target_length]

    # Convert to PyTorch tensor for neural network processing
    mfcc_tensor = torch.tensor(
        combined_mfcc,
        dtype=torch.float32,
    )
    print(
    f"Raw MFCC statistics | "
    f"mean={mfcc_tensor.mean().item():.4f}, "
    f"std={mfcc_tensor.std().item():.4f}, "
    f"min={mfcc_tensor.min().item():.4f}, "
    f"max={mfcc_tensor.max().item():.4f}"
)
    # Add batch dimension: (26, 128) -> (1, 26, 128) for model input
    return mfcc_tensor.unsqueeze(0)

