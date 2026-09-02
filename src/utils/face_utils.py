import torch

# FACE UTTERANCE HELPERS

def get_face_classifier(model):
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
    model_output = model(
        images,
        return_embeddings=True,
    )

    if isinstance(model_output, torch.Tensor):
        embeddings = model_output

    elif isinstance(model_output, (tuple, list)):
        if len(model_output) < 2:
            raise ValueError(
                "Face model returned a tuple/list without embeddings."
            )

        embeddings = model_output[-1]

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
    Average valid frame embeddings for each utterance and use the
    averaged embedding to generate face logits.
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