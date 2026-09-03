# Dynamic Modality Weighting for Robust Multimodal Emotion Recognition under Conflicting Signals

This repository contains the implementation of my multimodal emotion recognition project. The system combines **speech** and **facial expression** information to classify emotions into four categories:

- Angry
- Happy
- Neutral
- Sad

The main focus of the project is not only to build an emotion recognition model, but to investigate how speech and facial information should be combined when both modalities are not equally reliable. In real situations, a person's face and voice may not always communicate the same emotional cue. For example, a person may appear happy while speaking in an angry tone, or a facial image may be unclear while the speech signal is still useful.

To study this problem, the project compares **fixed 50/50 modality weighting** with an **entropy-guided dynamic weighting strategy**.

---

## Project Aim

The aim of this project is to develop and evaluate a multimodal emotion recognition framework that combines speech and facial information using modality weighting.

The project investigates whether dynamically adjusting the contribution of each modality can improve emotion recognition when the reliability of speech and facial cues varies.

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Pre-trained models in `checkpoints/` folder (see QUICK_START.md)

### Installation Steps

1. **Clone the repository**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

For detailed setup instructions and deployment options, see [QUICK_START.md](QUICK_START.md).

---

## Project Overview

The system is built around three main components:

1. **Speech Emotion Recognition Model**
   - Uses audio data from IEMOCAP.
   - Extracts MFCC features from speech recordings.
   - Trains a CNN-based speech emotion recognition model.

2. **Facial Emotion Recognition Model**
   - Uses FER2013 for facial model pretraining.
   - Extracts facial frames from IEMOCAP videos.
   - Fine-tunes the facial model on IEMOCAP facial images.

3. **Multimodal Fusion Model**
   - Combines speech and facial embeddings.
   - Applies modality weighting before final classification.
   - Compares fixed equal weighting with entropy-guided dynamic weighting.

---

## Emotion Classes

The final emotion categories used in this project are:

```text
angry
happy
neutral
sad
```

For IEMOCAP, the original emotion labels were mapped into these four classes. The `excited` label was merged with `happy`, while labels outside the selected categories were excluded.

---

## Fusion Strategy

This project compares two main fusion strategies.

### 1. Fixed Weighting Baseline

The fixed baseline gives equal importance to speech and facial information:

```text
speech weight = 0.5
face weight   = 0.5
```

This provides a simple baseline for checking whether a more adaptive fusion method is actually useful.

### 2. Entropy-Guided Dynamic Weighting

The dynamic weighting approach estimates the confidence of each modality using entropy. A modality with lower uncertainty receives a higher contribution during fusion.

The weighting is calculated as:

```text
w_s = c_s / (c_s + c_f)
w_f = c_f / (c_s + c_f)
```

Where:

```text
c_s = confidence of the speech model
c_f = confidence of the face model
w_s = speech weight
w_f = face weight
```

The weighted embeddings are then combined and passed into the fusion classifier for final emotion prediction.

---

## System Pipeline

```text
IEMOCAP Audio
     |
     v
MFCC Feature Extraction
     |
     v
Speech CNN Model
     |
     v
Speech Embedding
     |
     |----------------------------------|
                                        |
                                        v
                              Modality Weighting
                                        |
                                        v
                              Weighted Fusion Layer
                                        |
                                        v
                              Final Emotion Prediction


IEMOCAP Video
     |
     v
Facial Frame Extraction
     |
     v
Face Model
     |
     v
Face Embedding
     |
     |----------------------------------|
```

---

## Repository Structure

```text
.
├── app.py
├── .gitignore
├── .streamlit_config.toml
├── requirements_streamlit.txt
├── FRONTEND_OVERVIEW.md
├── MATHEMATICS.md
├── QUICK_START.md
│
├── scripts/
│   ├── evaluate.py
│   ├── finetune_face_iemocap.py
│   ├── pretrain_face_fer2013.py
│   ├── train_multimodal_fusion.py
│   └── train_speech.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   └── iemocap_face_extractor.py
│   │
│   ├── models/
│   │   ├── embedding_fusion.py
│   │   ├── face_model.py
│   │   └── speech_model.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── confidence.py
│       ├── face_utils.py
│       ├── iemocap_dataloader.py
│       ├── iemocap_extracted_faces_dataset.py
│       ├── iemocap_multimodal_dataset.py
│       ├── metrics.py
│       ├── mlflow_tracking.py
│       ├── streamlit_utils.py
│       └── visualization.py
```

---

## Files Not Included

The following files and folders are intentionally excluded from this repository:

```text
mer/
data/
checkpoints/
__pycache__/
*.pth
*.pt
*.ckpt
```

These files are excluded because they contain virtual environment files, raw datasets, processed data, cached files, or trained model checkpoints. The repository is kept focused on the source code and project structure.

---

## Datasets Used

### IEMOCAP

IEMOCAP was used as the main dataset for speech emotion recognition and multimodal evaluation. Audio files were used for MFCC extraction, while video files were used to extract facial images for the facial model.

### FER2013

FER2013 was used to pretrain the facial emotion recognition model before fine-tuning it on facial images extracted from IEMOCAP.

The datasets are not included in this repository. They must be downloaded separately according to their official access conditions.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Ekeminimbabah/Dynamic-Modality-Weighting-for-Robust-Multimodal-Emotions-Recognition-Under-Conflicting-Signal.git
```

Move into the project folder:

```bash
cd Dynamic-Modality-Weighting-for-Robust-Multimodal-Emotions-Recognition-Under-Conflicting-Signal
```

Create a virtual environment:

```bash
python -m venv mer
```

Activate the virtual environment:

```bash
# Windows Command Prompt
mer\Scripts\activate

# Git Bash
source mer/Scripts/activate
```

Install the required packages:

```bash
pip install -r requirements_streamlit.txt
```

---

## Running the Streamlit Application

This project includes a Streamlit interface for demonstrating the emotion recognition system.

Run:

```bash
streamlit run app.py
```

---

## Training Workflow

The project was developed in stages.

### 1. Pretrain the Facial Model on FER2013

```bash
python scripts/pretrain_face_fer2013.py
```

### 2. Extract Facial Frames from IEMOCAP

```bash
python src/data/iemocap_face_extractor.py
```

### 3. Fine-tune the Facial Model on IEMOCAP

```bash
python scripts/finetune_face_iemocap.py
```

### 4. Train the Speech Emotion Recognition Model

```bash
python scripts/train_speech.py
```

### 5. Train the Multimodal Fusion Model

```bash
python scripts/train_multimodal_fusion.py
```

### 6. Evaluate the Models

```bash
python scripts/evaluate.py
```

---

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curves
- Precision-recall curves

The evaluation compared:

1. Speech-only model
2. Face-only model
3. Multimodal fusion model
4. Fixed 50/50 weighting baseline
5. Entropy-guided dynamic weighting

The results showed that combining speech and facial information improved emotion recognition compared with using either modality alone. The comparison between fixed and dynamic weighting also showed the importance of testing fusion strategies empirically rather than assuming that a more complex method will always perform better.

---

## Summary of Results

The main model comparison showed the following performance:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Speech model | 49.1% | 50.1% | 49.1% | 48.0% |
| Face model | 63.8% | 64.9% | 63.8% | 63.6% |
| Multimodal fusion model | 73.1% | 74.1% | 73.1% | 72.8% |

These results show that the multimodal framework produced stronger performance than the individual speech and facial models.

---

## Key Features

- Speech emotion recognition using MFCC features.
- Facial emotion recognition using FER2013 pretraining.
- IEMOCAP facial frame extraction.
- Fine-tuning of the facial model on IEMOCAP.
- Multimodal fusion of speech and facial embeddings.
- Fixed 50/50 weighting baseline.
- Entropy-based confidence estimation.
- Dynamic modality weighting.
- Streamlit demonstration interface.
- Evaluation using standard classification metrics.

---

## Research Focus

This project focuses on the problem of conflicting or unreliable emotional cues in multimodal emotion recognition.

Instead of assuming that speech and facial information should always contribute equally, the project explores whether confidence-based weighting can help the model adjust the contribution of each modality. This is important because speech and facial expressions may carry different emotional signals, especially in real-world conditions.

---

## Limitations

This study has some limitations. The evaluation was mainly based on the IEMOCAP dataset, with FER2013 used only for facial model pretraining, so the findings may not generalise fully to other datasets or real-world environments. Class imbalance also affected model training and required class weighting to reduce prediction bias. In addition, the framework was tested under controlled experimental conditions rather than live audio-video settings, meaning further real-world testing is needed.

---

## Author

**Ekemini Mbabah**

Project area: Multimodal Emotion Recognition  
Main focus: Dynamic Modality Weighting under Conflicting Signals
