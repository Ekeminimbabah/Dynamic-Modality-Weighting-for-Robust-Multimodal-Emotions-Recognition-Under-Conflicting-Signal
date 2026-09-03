"""
Multimodal Embedding Fusion Module
Combines speech and face embeddings using attention-based weighting
"""

import torch
import torch.nn as nn


class EmbeddingFusion(nn.Module):
    """
    Fuses speech and face embeddings using attention-based dynamic weighting
    
    Process:
    1. Takes embeddings from speech (128D) and face (128D) models
    2. Computes attention weights based on embedding importance
    3. Fuses embeddings using computed weights
    4. Produces final emotion classification (4 classes)
    
    Input: 
    - speech_embeddings: (batch_size, 128)
    - face_embeddings: (batch_size, 128)
    
    Output:
    - logits: (batch_size, 4) emotion predictions
    """
    
    def __init__(self, embedding_dim=128, num_emotions=4):
        super(EmbeddingFusion, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions
        
        # ===== ATTENTION WEIGHTS =====
        # Learned gating networks that compute per-sample modality importance.
        # Each network takes an embedding and outputs a scalar weight (0-1) via sigmoid.
        # This allows the model to adaptively adjust modality contribution based on input.
        self.speech_gate = nn.Sequential(
            # Compress embedding to decision representation
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            # Output single weight value for this modality
            nn.Linear(64, 1),
            nn.Sigmoid()  # Bound weight to [0, 1]
        )
        
        self.face_gate = nn.Sequential(
            # Same architecture ensures symmetric treatment of modalities
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # ===== FUSION LAYERS =====
        # Process concatenated weighted embeddings to learn cross-modal interactions.
        # This allows the model to learn how to best combine modalities beyond simple scaling.
        self.fusion = nn.Sequential(
            # Project from concatenated embeddings (256D) to common representation (256D)
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Further compress to 128D before final classification
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # ===== FINAL CLASSIFIER =====
        # Maps fused representation to emotion logits for classification
        self.classifier = nn.Linear(128, num_emotions)
    
    def forward(self, speech_embeddings, face_embeddings):
        """
        Fuse embeddings and produce emotion prediction
        
        Args:
            speech_embeddings (torch.Tensor): Speech features (batch_size, 128)
            face_embeddings (torch.Tensor): Face features (batch_size, 128)
        
        Returns:
            dict: Contains:
                - 'logits': (batch_size, 4) emotion predictions
                - 'speech_weight': attention weight for speech
                - 'face_weight': attention weight for face
                - 'fused_embeddings': (batch_size, 128) fused features
        """
        
        # ===== COMPUTE ATTENTION WEIGHTS =====
        # Each gating network independently computes modality importance from its embedding.
        # This is different from fixed 0.5/0.5 - each sample gets adaptive weights.
        speech_weight = self.speech_gate(speech_embeddings)  # (batch_size, 1)
        face_weight = self.face_gate(face_embeddings)  # (batch_size, 1)
        
        # Normalize weights so they sum to 1: ensures combination is a proper weighted average
        # Both modalities always contribute, but their relative importance varies per sample
        total_weight = speech_weight + face_weight
        speech_weight_norm = speech_weight / (total_weight + 1e-8)
        face_weight_norm = face_weight / (total_weight + 1e-8)
        
        # ===== APPLY WEIGHTS =====
        # Scale embeddings by computed importance: high weight preserves embedding direction,
        # low weight dampens contribution. Trained end-to-end with emotion classification loss.
        weighted_speech = speech_embeddings * speech_weight_norm
        weighted_face = face_embeddings * face_weight_norm
        
        # ===== FUSE EMBEDDINGS =====
        # Concatenate scaled representations: combines modality strengths while maintaining
        # their relative contributions determined by learned gating functions
        fused = torch.cat([weighted_speech, weighted_face], dim=1)
        
        # Process through fusion layers: allows model to learn interaction between weighted modalities
        # beyond simple concatenation (e.g., resolving conflicts when modalities disagree)
        fused_features = self.fusion(fused)
        
        # ===== CLASSIFY =====
        # Predict emotion from learned multimodal representation
        logits = self.classifier(fused_features)
        
        # Return both predictions and intermediate representations for analysis and debugging
        return {
            'logits': logits,
            'speech_weight': speech_weight_norm.squeeze(-1),
            'face_weight': face_weight_norm.squeeze(-1),
            'fused_embeddings': fused_features
        }
