"""
Multimodal Emotion Recognition - Research Evaluation Interface
MSc Research Prototype for evaluating dynamic multimodal fusion models
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
import cv2
from PIL import Image as PILImage, ImageFilter
import librosa
from torchvision import transforms

st.set_page_config(
    page_title="Emotion Recognition - Research Evaluation",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f0f4f8; }
.emotion-card {
    padding: 20px; border-radius: 15px; color: white; 
    text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.emotion-angry { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.emotion-happy { background: linear-gradient(135deg, #ffd89b 0%, #ff9d56 100%); }
.emotion-neutral { background: linear-gradient(135deg, #a8e6cf 0%, #56ab91 100%); }
.emotion-sad { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.metric-box {
    background: #e8f4f8; padding: 12px; border-radius: 8px; margin: 5px 0;
    border-left: 4px solid #0084d4;
}
</style>
""", unsafe_allow_html=True)

from src.models.face_model import FaceEmotionModel
from src.models.speech_model import SpeechEmotionModel
from src.models.embedding_fusion import EmbeddingFusion
from src.utils.confidence import get_entropy_based_confidence

# ===== SETUP =====
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTIONS = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
EMOJIS = {0: '😠', 1: '😊', 2: '😐', 3: '😢'}
NUM_EMOTIONS = 4


@st.cache_resource
def load_models():
    """Load all trained models: face, speech, and both fusion models"""
    
    # Face model
    face_model = FaceEmotionModel(num_emotions=NUM_EMOTIONS).to(DEVICE)
    face_ckpt = PROJECT_ROOT / "checkpoints" / "face_iemocap.pth"
    if not face_ckpt.exists():
        return None, None, None, None, f"Face checkpoint missing: {face_ckpt}"
    face_model.load_state_dict(torch.load(face_ckpt, map_location=DEVICE))
    face_model.eval()
    
    # Speech model
    speech_model = SpeechEmotionModel(num_emotions=NUM_EMOTIONS).to(DEVICE)
    speech_ckpt = PROJECT_ROOT / "checkpoints" / "speech_iemocap.pth"
    if not speech_ckpt.exists():
        return None, None, None, None, f"Speech checkpoint missing: {speech_ckpt}"
    speech_model.load_state_dict(torch.load(speech_ckpt, map_location=DEVICE))
    speech_model.eval()
    
    # Fixed fusion model (equal weights baseline)
    fusion_equal = EmbeddingFusion(embedding_dim=128, num_emotions=NUM_EMOTIONS).to(DEVICE)
    fusion_equal_ckpt = PROJECT_ROOT / "checkpoints" / "fusion_model_equal.pth"
    if not fusion_equal_ckpt.exists():
        return None, None, None, None, f"Fixed fusion checkpoint missing: {fusion_equal_ckpt}"
    fusion_equal.load_state_dict(torch.load(fusion_equal_ckpt, map_location=DEVICE))
    fusion_equal.eval()
    
    # Dynamic fusion model (entropy-based)
    fusion_dynamic = EmbeddingFusion(embedding_dim=128, num_emotions=NUM_EMOTIONS).to(DEVICE)
    fusion_dynamic_ckpt = PROJECT_ROOT / "checkpoints" / "fusion_model_dynamic.pth"
    if not fusion_dynamic_ckpt.exists():
        return None, None, None, None, f"Dynamic fusion checkpoint missing: {fusion_dynamic_ckpt}"
    fusion_dynamic.load_state_dict(torch.load(fusion_dynamic_ckpt, map_location=DEVICE))
    fusion_dynamic.eval()
    
    return face_model, speech_model, fusion_equal, fusion_dynamic, None


def preprocess_face_image(image):
    """
    Preprocess face image exactly as in training.
    Input: PIL Image or numpy array
    Output: (1, 1, 48, 48) tensor normalized to [-1, 1]
    """
    if isinstance(image, PILImage.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert RGB to grayscale
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Resize to 48x48
    img_resized = cv2.resize(img_gray, (48, 48))
    
    # Convert to PIL Image for transform pipeline
    img_pil = PILImage.fromarray(img_resized)
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # [-1, 1] range
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(DEVICE)


def preprocess_audio(audio_file):
    """
    Extract MFCC features from audio exactly as in training.
    Output: (1, 1, 26, 128) tensor [batch, channels, n_mfcc, time_steps]
    """
    try:
        # Read audio bytes
        audio_bytes = audio_file.read()
        
        # Save temporarily
        ext = audio_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
            
            # Extract MFCC (13 coefficients)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13,
                n_fft=400,
                hop_length=160
            )
            
            # Add delta features
            delta_mfcc = librosa.feature.delta(mfcc)
            
            # Stack: (26, time_steps)
            combined_mfcc = np.vstack((mfcc, delta_mfcc))
            
            # Pad or trim to 128 time steps
            if combined_mfcc.shape[1] < 128:
                combined_mfcc = np.pad(
                    combined_mfcc,
                    ((0, 0), (0, 128 - combined_mfcc.shape[1])),
                    mode='constant'
                )
            else:
                combined_mfcc = combined_mfcc[:, :128]
            
            # Convert to tensor
            # Shape progression: (26, 128) -> (1, 26, 128) -> (1, 1, 26, 128)
            mfcc_tensor = torch.from_numpy(combined_mfcc).float()  # (26, 128)
            mfcc_tensor = mfcc_tensor.unsqueeze(0)  # Add channel: (1, 26, 128)
            mfcc_tensor = mfcc_tensor.unsqueeze(0)  # Add batch: (1, 1, 26, 128)
            return mfcc_tensor.to(DEVICE), None
        
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
    
    except Exception as e:
        return None, str(e)


def apply_face_blur(image, radius=5):
    """Apply Gaussian blur to PIL Image for robustness testing"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_gaussian_noise(audio, snr_db=10, seed=42):
    """Add reproducible Gaussian noise to waveform at specified SNR"""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(audio ** 2)
    
    if signal_power == 0:
        return audio
    
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape)
    noisy_audio = audio + noise
    
    return np.clip(noisy_audio, -1.0, 1.0)


def extract_mfcc_from_waveform(audio, sr=16000):
    """Extract MFCC features from audio waveform (not file path)"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=400, hop_length=160)
    delta_mfcc = librosa.feature.delta(mfcc)
    combined_mfcc = np.vstack((mfcc, delta_mfcc))
    
    if combined_mfcc.shape[1] < 128:
        combined_mfcc = np.pad(combined_mfcc, ((0, 0), (0, 128 - combined_mfcc.shape[1])), mode='constant')
    else:
        combined_mfcc = combined_mfcc[:, :128]
    
    mfcc_tensor = torch.from_numpy(combined_mfcc).float()
    mfcc_tensor = mfcc_tensor.unsqueeze(0).unsqueeze(0)
    return mfcc_tensor.to(DEVICE)


def evaluate_pair(face_image, audio_waveform, face_model, speech_model, fusion_equal, fusion_dynamic):
    """Unified evaluation: returns structured results for a face image and audio waveform"""
    face_tensor = preprocess_face_image(face_image)
    speech_tensor = extract_mfcc_from_waveform(audio_waveform)
    
    face_logits, face_embeddings = get_face_logits_and_embeddings(face_model, face_tensor)
    speech_logits, speech_embeddings = get_speech_logits_and_embeddings(speech_model, speech_tensor)
    
    fixed_probs, fixed_meta = predict_with_fusion(speech_embeddings, face_embeddings, speech_logits, face_logits, fusion_equal, mode="equal")
    dynamic_probs, dynamic_meta = predict_with_fusion(speech_embeddings, face_embeddings, speech_logits, face_logits, fusion_dynamic, mode="dynamic")
    
    speech_pred = torch.argmax(speech_logits, dim=1)[0].item()
    face_pred = torch.argmax(face_logits, dim=1)[0].item()
    
    speech_conf = torch.softmax(speech_logits, dim=1)[0][speech_pred].item()
    face_conf = torch.softmax(face_logits, dim=1)[0][face_pred].item()
    
    fixed_pred = np.argmax(fixed_probs)
    dynamic_pred = np.argmax(dynamic_probs)
    
    return {
        'speech_pred': speech_pred,
        'face_pred': face_pred,
        'speech_conf': speech_conf,
        'face_conf': face_conf,
        'speech_entropy': dynamic_meta['entropy_speech'],
        'face_entropy': dynamic_meta['entropy_face'],
        'speech_weight': dynamic_meta['w_speech'],
        'face_weight': dynamic_meta['w_face'],
        'fixed_pred': fixed_pred,
        'fixed_conf': fixed_probs[fixed_pred],
        'dynamic_pred': dynamic_pred,
        'dynamic_conf': dynamic_probs[dynamic_pred]
    }


def get_face_logits_and_embeddings(face_model, face_tensor):
    """Get both logits and embeddings from face model"""
    with torch.no_grad():
        embeddings = face_model(face_tensor, return_embeddings=True)
        logits = face_model(face_tensor, return_embeddings=False)
    return logits, embeddings


def get_speech_logits_and_embeddings(speech_model, speech_tensor):
    """Get both logits and embeddings from speech model"""
    with torch.no_grad():
        embeddings = speech_model(speech_tensor, return_embeddings=True)
        logits = speech_model(speech_tensor, return_embeddings=False)
    return logits, embeddings


def calculate_entropy_based_confidence(logits, num_classes=NUM_EMOTIONS):
    """
    Calculate entropy-based confidence: c = 1 - H(p) / log(C)
    where H(p) = -Σ p_i * log(p_i)
    """
    confidence = get_entropy_based_confidence(logits, num_classes)
    return confidence[0]  # Single sample


def calculate_entropy(logits, num_classes=NUM_EMOTIONS):
    """Calculate raw entropy H(p) = -Σ p_i * log(p_i)"""
    softmax = torch.softmax(logits, dim=1)
    entropy = -torch.sum(softmax * torch.log(softmax + 1e-8), dim=1)
    return entropy[0].item()


def apply_dynamic_weighting(speech_logits, face_logits):
    """
    Apply entropy-based dynamic weighting to modality logits.
    
    Returns:
        w_speech: normalized speech weight
        w_face: normalized face weight
        entropy_speech: raw entropy of speech predictions
        entropy_face: raw entropy of face predictions
        conf_speech: confidence of speech (1 - H/log(C))
        conf_face: confidence of face (1 - H/log(C))
    """
    conf_speech = calculate_entropy_based_confidence(speech_logits)
    conf_face = calculate_entropy_based_confidence(face_logits)
    
    entropy_speech = calculate_entropy(speech_logits)
    entropy_face = calculate_entropy(face_logits)
    
    # Normalize weights
    conf_sum = conf_speech + conf_face + 1e-8
    w_speech = conf_speech / conf_sum
    w_face = conf_face / conf_sum
    
    return w_speech, w_face, entropy_speech, entropy_face, conf_speech, conf_face


def predict_with_fusion(
    speech_embeddings, face_embeddings,
    speech_logits, face_logits,
    fusion_model, mode="dynamic"
):
    """
    Run fusion model with embeddings.
    
    For dynamic mode, apply entropy-based weighting to embeddings first.
    For equal mode, apply fixed 0.5/0.5 weighting.
    """
    
    if mode == "dynamic":
        w_speech, w_face, h_speech, h_face, c_speech, c_face = apply_dynamic_weighting(
            speech_logits, face_logits
        )
        weighted_speech = w_speech * speech_embeddings
        weighted_face = w_face * face_embeddings
        meta = {
            'w_speech': w_speech.item(),
            'w_face': w_face.item(),
            'entropy_speech': h_speech,
            'entropy_face': h_face,
            'conf_speech': c_speech.item(),
            'conf_face': c_face.item()
        }
    else:  # equal
        w_speech = 0.5
        w_face = 0.5
        weighted_speech = w_speech * speech_embeddings
        weighted_face = w_face * face_embeddings
        meta = {
            'w_speech': w_speech,
            'w_face': w_face,
            'entropy_speech': None,
            'entropy_face': None,
            'conf_speech': None,
            'conf_face': None
        }
    
    with torch.no_grad():
        fusion_output = fusion_model(weighted_speech, weighted_face)
        logits = fusion_output['logits']
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    return probs, meta


def display_emotion_prediction(logits, title="Prediction"):
    """Display emotion prediction from logits with bar chart and card"""
    probs = torch.softmax(logits, dim=1)[0].cpu().detach().numpy()
    display_emotion_prediction_probs(probs, title)


def display_emotion_prediction_probs(probs, title="Prediction"):
    """Display emotion prediction from already-calculated probabilities (no Softmax)"""
    pred_idx = np.argmax(probs)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        import plotly.graph_objects as go
        emotions_list = [EMOTIONS[i] for i in range(NUM_EMOTIONS)]
        probs_pct = probs * 100
        
        fig = go.Figure([go.Bar(
            y=emotions_list, x=probs_pct, orientation='h',
            marker=dict(
                color=probs_pct,
                colorscale=[[0,'#56ab91'], [0.33,'#764ba2'], [0.66,'#ff9d56'], [1,'#f5576c']]
            ),
            text=[f'{p:.1f}%' for p in probs_pct],
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Confidence (%)",
            yaxis_title="Emotion",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="emotion-card emotion-{EMOTIONS[pred_idx].lower()}">
            <div style="font-size: 3em; margin: 10px 0;">{EMOJIS[pred_idx]}</div>
            <div style="font-size: 1.3em; font-weight: bold;">{EMOTIONS[pred_idx]}</div>
            <div style="font-size: 1.8em; font-weight: bold;">{probs[pred_idx] * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    return pred_idx


# ===== MAIN UI =====
st.title("🧠 Multimodal Emotion Recognition")
st.markdown("**Research Evaluation Interface**")
st.info(
    "ℹ️ This interface is an MSc research prototype for evaluating trained "
    "emotion-recognition models. It is not a clinical diagnostic system."
)

# Load all models
with st.spinner("Loading models..."):
    face_model, speech_model, fusion_equal, fusion_dynamic, error_msg = load_models()

if error_msg:
    st.error(f"❌ Model Loading Error:\n{error_msg}")
    st.stop()

st.success("✓ All models loaded successfully")

# ===== FILE UPLOADS =====
st.markdown("---")
st.subheader("📂 Input Files")
col_img, col_aud = st.columns(2)

uploaded_image = None
uploaded_audio = None

with col_img:
    st.markdown("**📸 Facial Image**")
    uploaded_image = st.file_uploader(
        "Upload face image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="img"
    )
    if uploaded_image:
        image = PILImage.open(uploaded_image)
        st.image(image, caption="Uploaded face image", width=250)

with col_aud:
    st.markdown("**🎤 Speech Recording**")
    uploaded_audio = st.file_uploader(
        "Upload audio (WAV, MP3, M4A)",
        type=["wav", "mp3", "m4a"],
        key="aud"
    )
    if uploaded_audio:
        st.audio(uploaded_audio)

# Optional expected emotion for validation
st.markdown("---")
expected_emotion = st.selectbox(
    "**Optional: Known / Expected Emotion** (for validation only)",
    ["Not specified", "Angry", "Happy", "Neutral", "Sad"]
)

# ===== RUN EVALUATION =====
if st.button("▶ Run Evaluation", use_container_width=True, type="primary"):
    if not uploaded_image or not uploaded_audio:
        st.error("❌ Please upload both image AND audio files")
        st.stop()
    
    st.markdown("---")
    st.subheader("🔄 Processing...")
    
    # Preprocess inputs
    progress = st.progress(0)
    
    try:
        # Face preprocessing
        progress.progress(20)
        face_tensor = preprocess_face_image(PILImage.open(uploaded_image))
        
        # Speech preprocessing
        progress.progress(40)
        speech_tensor, audio_error = preprocess_audio(uploaded_audio)
        if audio_error:
            st.error(f"❌ Audio preprocessing failed: {audio_error}")
            st.stop()
        
        # Face inference
        progress.progress(50)
        face_logits, face_embeddings = get_face_logits_and_embeddings(face_model, face_tensor)
        
        # Speech inference
        progress.progress(60)
        speech_logits, speech_embeddings = get_speech_logits_and_embeddings(speech_model, speech_tensor)
        
        # Fixed fusion
        progress.progress(75)
        fixed_probs, fixed_meta = predict_with_fusion(
            speech_embeddings, face_embeddings,
            speech_logits, face_logits,
            fusion_equal, mode="equal"
        )
        
        # Dynamic fusion
        progress.progress(90)
        dynamic_probs, dynamic_meta = predict_with_fusion(
            speech_embeddings, face_embeddings,
            speech_logits, face_logits,
            fusion_dynamic, mode="dynamic"
        )
        
        progress.progress(100)
        progress.empty()
        
        # ===== DISPLAY RESULTS =====
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Comparison table
        st.markdown("**Model Comparison**")
        face_pred = torch.argmax(face_logits, dim=1)[0].item()
        speech_pred = torch.argmax(speech_logits, dim=1)[0].item()
        fixed_pred = np.argmax(fixed_probs)
        dynamic_pred = np.argmax(dynamic_probs)
        
        face_conf = torch.softmax(face_logits, dim=1)[0][face_pred].item()
        speech_conf = torch.softmax(speech_logits, dim=1)[0][speech_pred].item()
        fixed_conf = fixed_probs[fixed_pred]
        dynamic_conf = dynamic_probs[dynamic_pred]
        
        results_data = {
            'Model': ['Speech', 'Face', 'Fixed 50/50 Fusion', 'Dynamic Entropy Fusion'],
            'Predicted Emotion': [
                EMOTIONS[speech_pred],
                EMOTIONS[face_pred],
                EMOTIONS[fixed_pred],
                EMOTIONS[dynamic_pred]
            ],
            'Confidence': [
                f'{speech_conf:.2%}',
                f'{face_conf:.2%}',
                f'{fixed_conf:.2%}',
                f'{dynamic_conf:.2%}'
            ]
        }
        
        st.dataframe(results_data, use_container_width=True, hide_index=True)
        
        # Validation against expected emotion
        if expected_emotion != "Not specified":
            st.markdown("---")
            st.subheader("✓ Validation")
            expected_idx = {"Angry": 0, "Happy": 1, "Neutral": 2, "Sad": 3}[expected_emotion]
            
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            models = ["Speech", "Face", "Fixed 50/50", "Dynamic"]
            preds = [speech_pred, face_pred, fixed_pred, dynamic_pred]
            
            for col, model, pred in zip(cols, models, preds):
                with col:
                    is_correct = pred == expected_idx
                    status = "✓ Correct" if is_correct else "✗ Incorrect"
                    color = "green" if is_correct else "red"
                    st.markdown(f"**{model}**\n`{status}`", unsafe_allow_html=True)
        
        # Detailed results for each model
        st.markdown("---")
        st.subheader("🔍 Detailed Model Outputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Speech Model**")
            display_emotion_prediction(speech_logits, "Speech Prediction")
        
        with col2:
            st.markdown("**Facial Model**")
            display_emotion_prediction(face_logits, "Face Prediction")
        
        # ===== FUSION COMPARISON =====
        st.markdown("---")
        st.subheader("⚖️ Fusion Comparison")
        
        fusion_col1, fusion_col2 = st.columns(2)
        
        with fusion_col1:
            st.markdown("**Fixed 50/50 Fusion**")
            st.markdown("""
            <div class="metric-box">
            <strong>Speech Weight:</strong> 0.50<br>
            <strong>Face Weight:</strong> 0.50<br>
            </div>
            """, unsafe_allow_html=True)
            display_emotion_prediction_probs(fixed_probs, "Fixed Fusion Prediction")
        
        with fusion_col2:
            st.markdown("**Dynamic Entropy-Based Fusion**")
            st.markdown(f"""
            <div class="metric-box">
            <strong>Speech Weight:</strong> {dynamic_meta['w_speech']:.3f}<br>
            <strong>Face Weight:</strong> {dynamic_meta['w_face']:.3f}<br>
            <strong>Weight Sum:</strong> {dynamic_meta['w_speech'] + dynamic_meta['w_face']:.3f}
            </div>
            """, unsafe_allow_html=True)
            display_emotion_prediction_probs(dynamic_probs, "Dynamic Fusion Prediction")
        
        # ===== DYNAMIC FUSION DETAILS =====
        st.markdown("---")
        st.subheader("📈 Dynamic Fusion - Entropy & Confidence Details")
        
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        
        with detail_col1:
            st.markdown("**Speech Entropy**")
            st.metric("H(p)", f"{dynamic_meta['entropy_speech']:.4f}")
        
        with detail_col2:
            st.markdown("**Face Entropy**")
            st.metric("H(p)", f"{dynamic_meta['entropy_face']:.4f}")
        
        with detail_col3:
            st.markdown("**Speech Confidence**")
            st.metric("c", f"{dynamic_meta['conf_speech']:.4f}")
        
        with detail_col4:
            st.markdown("**Face Confidence**")
            st.metric("c", f"{dynamic_meta['conf_face']:.4f}")
        
        st.markdown("""
        **Explanation:**
        - **Entropy (H):** Measures prediction uncertainty. Lower = more confident.
        - **Confidence (c):** Normalized score 1 - H/log(4). Higher = model more certain about prediction.
        - **Weights:** Automatically normalized so they sum to 1. Modalities with higher confidence receive higher weight.
        """)
    
    except Exception as e:
        st.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ===== CONTROLLED ROBUSTNESS TESTS =====
st.markdown("---")
st.subheader("⚙️ Controlled Robustness Tests")
st.info(
    "ℹ️ These tests examine how trained fusion models respond when one modality is deliberately degraded. "
    "They are research tests and do not establish clinical robustness."
)

if not uploaded_image or not uploaded_audio:
    st.warning("Upload both image and audio files to run robustness tests.")
else:
    col_test1, col_test2 = st.columns(2)
    
    with col_test1:
        if st.button("📸 Compare Clean vs Blurred Face", use_container_width=True):
            try:
                # Load original audio waveform
                audio_bytes = uploaded_audio.read()
                uploaded_audio.seek(0)
                ext = uploaded_audio.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                try:
                    audio_clean, sr = librosa.load(tmp_path, sr=16000, mono=True)
                finally:
                    os.remove(tmp_path)
                
                # Prepare images
                face_clean = PILImage.open(uploaded_image)
                face_blurred = apply_face_blur(face_clean, radius=5)
                
                # Evaluate both
                result_clean = evaluate_pair(face_clean, audio_clean, face_model, speech_model, fusion_equal, fusion_dynamic)
                result_blur = evaluate_pair(face_blurred, audio_clean, face_model, speech_model, fusion_equal, fusion_dynamic)
                
                # Display comparison
                st.subheader("Face Image Robustness Test")
                st.markdown("**Blur Parameter:** radius = 5")
                
                # Build improved comparison table
                expected_idx = {"Angry": 0, "Happy": 1, "Neutral": 2, "Sad": 3}.get(expected_emotion, -1)
                
                comparison_data = {
                    'Condition': ['Clean', 'Blurred Face'],
                    'Speech Pred': [EMOTIONS[result_clean['speech_pred']], EMOTIONS[result_blur['speech_pred']]],
                    'Speech Conf': [f"{result_clean['speech_conf']:.2%}", f"{result_blur['speech_conf']:.2%}"],
                    'Face Pred': [EMOTIONS[result_clean['face_pred']], EMOTIONS[result_blur['face_pred']]],
                    'Face Conf': [f"{result_clean['face_conf']:.2%}", f"{result_blur['face_conf']:.2%}"],
                    'Speech Entropy': [f"{result_clean['speech_entropy']:.4f}", f"{result_blur['speech_entropy']:.4f}"],
                    'Face Entropy': [f"{result_clean['face_entropy']:.4f}", f"{result_blur['face_entropy']:.4f}"],
                    'Fixed S Weight': ['0.500', '0.500'],
                    'Fixed F Weight': ['0.500', '0.500'],
                    'Dynamic S Weight': [f"{result_clean['speech_weight']:.3f}", f"{result_blur['speech_weight']:.3f}"],
                    'Dynamic F Weight': [f"{result_clean['face_weight']:.3f}", f"{result_blur['face_weight']:.3f}"],
                    'Fixed Fusion Pred': [EMOTIONS[result_clean['fixed_pred']], EMOTIONS[result_blur['fixed_pred']]],
                    'Fixed Fusion Conf': [f"{result_clean['fixed_conf']:.2%}", f"{result_blur['fixed_conf']:.2%}"],
                    'Dynamic Fusion Pred': [EMOTIONS[result_clean['dynamic_pred']], EMOTIONS[result_blur['dynamic_pred']]],
                    'Dynamic Fusion Conf': [f"{result_clean['dynamic_conf']:.2%}", f"{result_blur['dynamic_conf']:.2%}"]
                }
                
                # Add validation columns if expected emotion specified
                if expected_emotion != "Not specified":
                    comparison_data['Fixed Correct?'] = [
                        'Yes' if result_clean['fixed_pred'] == expected_idx else 'No',
                        'Yes' if result_blur['fixed_pred'] == expected_idx else 'No'
                    ]
                    comparison_data['Dynamic Correct?'] = [
                        'Yes' if result_clean['dynamic_pred'] == expected_idx else 'No',
                        'Yes' if result_blur['dynamic_pred'] == expected_idx else 'No'
                    ]
                
                st.dataframe(comparison_data, use_container_width=True, hide_index=True)
            
            except Exception as e:
                st.error(f"Robustness test failed: {str(e)}")
    
    with col_test2:
        if st.button("🎤 Compare Clean vs Noisy Speech", use_container_width=True):
            try:
                # Load original audio waveform
                audio_bytes = uploaded_audio.read()
                uploaded_audio.seek(0)
                ext = uploaded_audio.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                try:
                    audio_clean, sr = librosa.load(tmp_path, sr=16000, mono=True)
                finally:
                    os.remove(tmp_path)
                
                # Prepare face
                face = PILImage.open(uploaded_image)
                
                # Add noise to audio
                audio_noisy = add_gaussian_noise(audio_clean, snr_db=10, seed=42)
                
                # Evaluate both
                result_clean = evaluate_pair(face, audio_clean, face_model, speech_model, fusion_equal, fusion_dynamic)
                result_noisy = evaluate_pair(face, audio_noisy, face_model, speech_model, fusion_equal, fusion_dynamic)
                
                # Display comparison
                st.subheader("Speech Robustness Test")
                st.markdown("**Noise Level:** 10 dB SNR | **Seed:** 42")
                
                # Build improved comparison table
                expected_idx = {"Angry": 0, "Happy": 1, "Neutral": 2, "Sad": 3}.get(expected_emotion, -1)
                
                comparison_data = {
                    'Condition': ['Clean', 'Noisy Speech (10 dB SNR)'],
                    'Speech Pred': [EMOTIONS[result_clean['speech_pred']], EMOTIONS[result_noisy['speech_pred']]],
                    'Speech Conf': [f"{result_clean['speech_conf']:.2%}", f"{result_noisy['speech_conf']:.2%}"],
                    'Face Pred': [EMOTIONS[result_clean['face_pred']], EMOTIONS[result_noisy['face_pred']]],
                    'Face Conf': [f"{result_clean['face_conf']:.2%}", f"{result_noisy['face_conf']:.2%}"],
                    'Speech Entropy': [f"{result_clean['speech_entropy']:.4f}", f"{result_noisy['speech_entropy']:.4f}"],
                    'Face Entropy': [f"{result_clean['face_entropy']:.4f}", f"{result_noisy['face_entropy']:.4f}"],
                    'Fixed S Weight': ['0.500', '0.500'],
                    'Fixed F Weight': ['0.500', '0.500'],
                    'Dynamic S Weight': [f"{result_clean['speech_weight']:.3f}", f"{result_noisy['speech_weight']:.3f}"],
                    'Dynamic F Weight': [f"{result_clean['face_weight']:.3f}", f"{result_noisy['face_weight']:.3f}"],
                    'Fixed Fusion Pred': [EMOTIONS[result_clean['fixed_pred']], EMOTIONS[result_noisy['fixed_pred']]],
                    'Fixed Fusion Conf': [f"{result_clean['fixed_conf']:.2%}", f"{result_noisy['fixed_conf']:.2%}"],
                    'Dynamic Fusion Pred': [EMOTIONS[result_clean['dynamic_pred']], EMOTIONS[result_noisy['dynamic_pred']]],
                    'Dynamic Fusion Conf': [f"{result_clean['dynamic_conf']:.2%}", f"{result_noisy['dynamic_conf']:.2%}"]
                }
                
                # Add validation columns if expected emotion specified
                if expected_emotion != "Not specified":
                    comparison_data['Fixed Correct?'] = [
                        'Yes' if result_clean['fixed_pred'] == expected_idx else 'No',
                        'Yes' if result_noisy['fixed_pred'] == expected_idx else 'No'
                    ]
                    comparison_data['Dynamic Correct?'] = [
                        'Yes' if result_clean['dynamic_pred'] == expected_idx else 'No',
                        'Yes' if result_noisy['dynamic_pred'] == expected_idx else 'No'
                    ]
                
                st.dataframe(comparison_data, use_container_width=True, hide_index=True)
            
            except Exception as e:
                st.error(f"Robustness test failed: {str(e)}")

st.markdown("---")
st.caption(
    "Multimodal Emotion Recognition Research Prototype | "
    "Emotions: Angry 😠 | Happy 😊 | Neutral 😐 | Sad 😢"
)
