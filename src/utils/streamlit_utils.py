"""
Simple prediction utilities for Streamlit app
Handles model loading, inference, and post-processing
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

EMOTION_NAMES = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
EMOTION_EMOJIS = {0: '😠', 1: '😊', 2: '😐', 3: '😢'}

class EmotionPredictor:
    """Unified emotion prediction interface"""
    
    def __init__(self, face_model, speech_model, fusion_model, device):
        self.face_model = face_model
        self.speech_model = speech_model
        self.fusion_model = fusion_model
        self.device = device
    
    def get_dominant_emotion(self, probs: np.ndarray) -> Tuple[str, float, str]:
        """Get the dominant emotion and its probability"""
        idx = np.argmax(probs)
        emotion_name = EMOTION_NAMES[idx]
        confidence = probs[idx]
        emoji = EMOTION_EMOJIS[idx]
        return emotion_name, confidence, emoji
    
    def format_results(self, probs: np.ndarray, modality: str) -> dict:
        """Format prediction results for display"""
        emotion_name, confidence, emoji = self.get_dominant_emotion(probs)
        
        return {
            'dominant_emotion': emotion_name,
            'confidence': confidence,
            'emoji': emoji,
            'modality': modality,
            'probabilities': {
                EMOTION_NAMES[i]: float(probs[i]) 
                for i in range(4)
            }
        }

def get_emotion_color(emotion_idx: int) -> str:
    """Get color for emotion visualization"""
    colors = {
        0: '#f5576c',   # Angry - Red
        1: '#ff9d56',   # Happy - Orange
        2: '#56ab91',   # Neutral - Green
        3: '#764ba2'    # Sad - Purple
    }
    return colors.get(emotion_idx, '#2196f3')

def format_confidence_display(confidence: float) -> str:
    """Format confidence score for display"""
    return f"{confidence * 100:.1f}%"
