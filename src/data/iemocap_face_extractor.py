
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


# CONFIGURATION
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}\n")

# Detect all faces, then choose the most confident one.
mtcnn = MTCNN(
    device=device,
    keep_all=True
)

EMOTION_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "neu": "neutral",
    "sad": "sad",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IEMOCAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IEMOCAP"
    / "IEMOCAP_full_release"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "IEMOCAP_faces"
)

FACE_CONFIDENCE_THRESHOLD = 0.90
MINIMUM_FACE_SIZE = 30

# Set this to True only when you want to delete old extracted faces.
CLEAR_OLD_OUTPUT = False

# PREPARE OUTPUT FOLDERS
if CLEAR_OLD_OUTPUT and OUTPUT_ROOT.exists():
    print(f"Removing old extracted faces from: {OUTPUT_ROOT}")
    shutil.rmtree(OUTPUT_ROOT)

for emotion_name in set(EMOTION_MAP.values()):
    (OUTPUT_ROOT / emotion_name).mkdir(
        parents=True,
        exist_ok=True
    )

print(f"IEMOCAP root: {IEMOCAP_ROOT}")
print(f"Output root: {OUTPUT_ROOT}\n")


# READ IEMOCAP ANNOTATIONS
def parse_emo_evaluation(file_path):
    """
    Read one IEMOCAP EmoEvaluation file.

    Returns
    -------
    list
        Each item contains:
        start time, end time, utterance ID and four-class label.
    """

    utterances = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith(("%", "C-", "A-")):
                continue

            try:
                if "[" not in line or "]" not in line:
                    continue

                time_text = line[
                    line.find("[") + 1:
                    line.find("]")
                ].strip()

                start_time, end_time = map(
                    float,
                    time_text.split("-")
                )

                remaining_text = line[
                    line.find("]") + 1:
                ].strip()

                parts = remaining_text.split()

                if len(parts) < 2:
                    continue

                utterance_id = parts[0]
                original_emotion = parts[1]

                if original_emotion not in EMOTION_MAP:
                    continue

                emotion_name = EMOTION_MAP[
                    original_emotion
                ]

                utterances.append(
                    (
                        start_time,
                        end_time,
                        utterance_id,
                        emotion_name,
                    )
                )

            except (ValueError, IndexError):
                continue

    return utterances


# EXTRACT FRAMES AT 20%, 50% AND 80%
def extract_utterance_frames(
    video_path,
    start_time,
    end_time
):
    """
    Extract frames from 20%, 50% and 80% of an utterance.

    Returns
    -------
    dict
        Example:
        {
            "start": RGB frame,
            "middle": RGB frame,
            "end": RGB frame
        }
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return {}

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        capture.release()
        return {}

    duration = end_time - start_time

    if duration <= 0:
        capture.release()
        return {}

    sample_times = {
        "start": start_time + (0.20 * duration),
        "middle": start_time + (0.50 * duration),
        "end": start_time + (0.80 * duration),
    }

    frames = {}

    for position, sample_time in sample_times.items():

        frame_number = int(sample_time * fps)

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = capture.read()

        if not success or frame is None:
            continue

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames[position] = frame_rgb

    capture.release()

    return frames


# FACE DETECTION AND QUALITY CHECK
def detect_valid_face(frame):
    """
    Detect the most confident face in one frame.

    Low-confidence, empty and very small crops are rejected.

    Returns
    -------
    numpy.ndarray or None
        A 224x224 RGB face image, or None.
    """

    try:
        pil_image = Image.fromarray(frame)

        boxes, probabilities = mtcnn.detect(
            pil_image,
            landmarks=False
        )

        if boxes is None or probabilities is None:
            return None

        valid_indices = [
            index
            for index, probability in enumerate(probabilities)
            if probability is not None
            and probability >= FACE_CONFIDENCE_THRESHOLD
        ]

        if not valid_indices:
            return None

        # Choose the detection with the highest confidence.
        best_index = max(
            valid_indices,
            key=lambda index: probabilities[index]
        )

        x_min, y_min, x_max, y_max = (
            boxes[best_index].astype(int)
        )

        frame_height, frame_width = frame.shape[:2]

        # Ensure coordinates remain inside the frame.
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame_width, x_max)
        y_max = min(frame_height, y_max)

        face_width = x_max - x_min
        face_height = y_max - y_min

        # Reject tiny detections.
        if (
            face_width < MINIMUM_FACE_SIZE
            or face_height < MINIMUM_FACE_SIZE
        ):
            return None

        face_crop = frame[
            y_min:y_max,
            x_min:x_max
        ]

        if face_crop.size == 0:
            return None

        face_resized = cv2.resize(
            face_crop,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        return face_resized

    except Exception:
        return None


# SAVE ONE UTTERANCE'S FACE IMAGES
def save_face(
    face_image,
    emotion_name,
    utterance_id,
    position
):
    """
    Save one face as start.png, middle.png or end.png.
    """

    utterance_folder = (
        OUTPUT_ROOT
        / emotion_name
        / utterance_id
    )

    utterance_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        utterance_folder
        / f"{position}.png"
    )

    face_bgr = cv2.cvtColor(
        face_image,
        cv2.COLOR_RGB2BGR
    )

    saved = cv2.imwrite(
        str(output_path),
        face_bgr
    )

    return saved

# MAIN EXTRACTION
print("=" * 70)
print("EXTRACTING IEMOCAP UTTERANCE-LEVEL FACE FRAMES")
print("=" * 70)

total_utterances = 0
utterances_with_faces = 0
utterances_skipped = 0
total_faces_saved = 0

saved_by_position = {
    "start": 0,
    "middle": 0,
    "end": 0,
}


for session_number in range(1, 6):

    print(f"\n{'=' * 70}")
    print(f"SESSION {session_number}")
    print(f"{'=' * 70}")

    session_path = (
        IEMOCAP_ROOT
        / f"Session{session_number}"
    )

    emotion_directory = (
        session_path
        / "dialog"
        / "EmoEvaluation"
    )

    video_directory = (
        session_path
        / "dialog"
        / "avi"
        / "DivX"
    )

    if not emotion_directory.exists():
        print("EmoEvaluation directory not found.")
        continue

    if not video_directory.exists():
        print("Video directory not found.")
        continue

    for emotion_file in sorted(
        emotion_directory.glob("*.txt")
    ):

        if emotion_file.name.startswith("._"):
            continue

        dialogue_id = emotion_file.stem

        video_path = (
            video_directory
            / f"{dialogue_id}.avi"
        )

        if not video_path.exists():
            print(
                f"  Video not found: "
                f"{video_path.name}"
            )
            continue

        print(f"\nProcessing: {dialogue_id}")

        utterances = parse_emo_evaluation(
            emotion_file
        )

        for (
            start_time,
            end_time,
            utterance_id,
            emotion_name,
        ) in utterances:

            total_utterances += 1

            frames = extract_utterance_frames(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time
            )

            if not frames:
                print(
                    f"  ✗ {utterance_id} "
                    f"({emotion_name}) — no frames"
                )

                utterances_skipped += 1
                continue

            saved_positions = []

            for position in (
                "start",
                "middle",
                "end"
            ):

                if position not in frames:
                    continue

                face_image = detect_valid_face(
                    frames[position]
                )

                if face_image is None:
                    continue

                was_saved = save_face(
                    face_image=face_image,
                    emotion_name=emotion_name,
                    utterance_id=utterance_id,
                    position=position
                )

                if was_saved:
                    saved_positions.append(position)
                    saved_by_position[position] += 1
                    total_faces_saved += 1

            # Keep an utterance when at least one valid face exists.
            if saved_positions:
                utterances_with_faces += 1

                print(
                    f"  ✓ {utterance_id} "
                    f"({emotion_name}) — "
                    f"{', '.join(saved_positions)}"
                )

            else:
                utterances_skipped += 1

                print(
                    f"  ✗ {utterance_id} "
                    f"({emotion_name}) — "
                    f"no valid faces"
                )


# SUMMARY
print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Total utterances processed: "
    f"{total_utterances}"
)

print(
    f"Utterances with at least one face: "
    f"{utterances_with_faces}"
)

print(
    f"Utterances skipped: "
    f"{utterances_skipped}"
)

print(
    f"Total face images saved: "
    f"{total_faces_saved}"
)

print(
    f"Start faces:  "
    f"{saved_by_position['start']}"
)

print(
    f"Middle faces: "
    f"{saved_by_position['middle']}"
)

print(
    f"End faces:    "
    f"{saved_by_position['end']}"
)

success_rate = (
    100 * utterances_with_faces
    / max(total_utterances, 1)
)

print(
    f"Utterance success rate: "
    f"{success_rate:.2f}%"
)

print(f"\nFaces saved to:\n{OUTPUT_ROOT}\n")