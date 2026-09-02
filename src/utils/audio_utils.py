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
    """

    audio, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=400,
        hop_length=160,
    )

    delta_mfcc = librosa.feature.delta(mfcc)

    combined_mfcc = np.vstack((mfcc, delta_mfcc))

    if combined_mfcc.shape[1] < target_length:
        padding = target_length - combined_mfcc.shape[1]

        combined_mfcc = np.pad(
            combined_mfcc,
            pad_width=((0, 0), (0, padding)),
            mode="constant",
        )
    else:
        combined_mfcc = combined_mfcc[:, :target_length]

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
    return mfcc_tensor.unsqueeze(0)

