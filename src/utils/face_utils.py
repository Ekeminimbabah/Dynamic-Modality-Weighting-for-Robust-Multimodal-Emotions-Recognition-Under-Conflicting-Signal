import torch

# FACE UTTERANCE HELPERS

def get_face_classifier(model):
    """
    Extract the emotion classification layer from FaceEmotionModel.
    
    Different model implementations may use different layer names (classifier, fc, etc).
    This utility handles flexible model architectures by checking common names.
    
    Args:
        model: FaceEmotionModel instance
    
    Returns:
        nn.Module: The final linear classification layer
    
    Raises:
        AttributeError: If no recognized classifier layer found
    """
    possible_names = (
        "classifier",
        "fc",
        "output_layer",
        "emotion_classifier",
    )

    for name in possible_names:
        classifier = getattr(model, name, None)

        if classifier is not None:
            return classifier

    raise AttributeError(
        "Could not find the final classification layer in "
        "FaceEmotionModel. Expected one of: classifier, fc, "
        "output_layer or emotion_classifier."
    )


def extract_face_embeddings(model, images):
    """
    Extract 128-dimensional embeddings from face images.
    
    Handles flexible model output formats (tensor, tuple, dict).
    Validates output shape to ensure compatibility with multimodal fusion.
    
    Args:
        model: FaceEmotionModel instance in eval mode
        images: Tensor of shape [batch_size, channels, height, width]
    
    Returns:
        torch.Tensor: Embeddings of shape [batch_size, 128]
    
    Raises:
        ValueError: If model output has unexpected structure
    """
    model_output = model(
        images,
        return_embeddings=True,
    )

    # Model may return embeddings directly as tensor
    if isinstance(model_output, torch.Tensor):
        embeddings = model_output

    # Or wrapped in tuple/list (logits, embeddings)
    elif isinstance(model_output, (tuple, list)):
        if len(model_output) < 2:
            raise ValueError(
                "Face model returned a tuple/list without embeddings."
            )

        embeddings = model_output[-1]

    # Or in dictionary with embedding key
    elif isinstance(model_output, dict):
        if "embeddings" in model_output:
            embeddings = model_output["embeddings"]

        elif "embedding" in model_output:
            embeddings = model_output["embedding"]

        else:
            raise KeyError(
                "Face model output dictionary has no embedding key."
            )

    else:
        raise TypeError(
            "Unsupported FaceEmotionModel output type."
        )

    # Validate shape: must be [batch, 128] for fusion compatibility
    if embeddings.ndim != 2:
        raise ValueError(
            "Expected frame embeddings with shape "
            "[number_of_frames, embedding_dimension], but received "
            f"{tuple(embeddings.shape)}."
        )

    return embeddings


def forward_face_utterance(
    model,
    frames,
    frame_mask,
):
    """
    Process multiple face frames from same utterance and generate utterance-level prediction.
    
    IEMOCAP utterances extract up to 3 frames (start, middle, end) for redundancy.
    This function averages embeddings across valid frames (masked) and uses the
    averaged representation to generate final face logits per utterance.
    
    Args:
        model: FaceEmotionModel instance
        frames: Tensor [batch, num_frames, channels, height, width] - up to 3 frames per utterance
        frame_mask: Tensor [batch, num_frames] - binary mask (1=frame exists, 0=missing)
    
    Returns:
        logits: Tensor [batch, 4] - emotion predictions per utterance
        utterance_embeddings: Tensor [batch, 128] - averaged face embeddings
    """

    if frames.ndim != 5:
        raise ValueError(
            "Expected face frames with shape "
            "[batch, frames, channels, height, width], but received "
            f"{tuple(frames.shape)}."
        )

    (
        batch_size,
        number_of_frames,
        channels,
        height,
        width,
    ) = frames.shape

    # Reshape for batch processing: treat multiple frames as separate samples
    # Process all frames together, then reshape back to utterance level
    flat_frames = frames.reshape(
        batch_size * number_of_frames,
        channels,
        height,
        width,
    )

    frame_embeddings = extract_face_embeddings(
        model,
        flat_frames,
    )

    embedding_dim = frame_embeddings.size(1)

    frame_embeddings = frame_embeddings.reshape(
        batch_size,
        number_of_frames,
        embedding_dim,
    )

    expanded_mask = frame_mask.unsqueeze(-1).to(
        device=frame_embeddings.device,
        dtype=frame_embeddings.dtype,
    )

    utterance_embeddings = (
        (frame_embeddings * expanded_mask).sum(dim=1)
        / expanded_mask.sum(dim=1).clamp(min=1.0)
    )

    classifier = get_face_classifier(model)

    face_logits = classifier(
        utterance_embeddings
    )

    return utterance_embeddings, face_logits