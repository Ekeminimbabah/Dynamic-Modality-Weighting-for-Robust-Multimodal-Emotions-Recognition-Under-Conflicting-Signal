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
        # Learn how important each modality is
        self.speech_gate = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Weight between 0 and 1
        )
        
        self.face_gate = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # ===== FUSION LAYERS =====
        # Fuse the weighted embeddings
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # ===== FINAL CLASSIFIER =====
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
        speech_weight = self.speech_gate(speech_embeddings)  # (batch_size, 1)
        face_weight = self.face_gate(face_embeddings)  # (batch_size, 1)
        
        # Normalize weights to sum to 1
        total_weight = speech_weight + face_weight
        speech_weight_norm = speech_weight / (total_weight + 1e-8)
        face_weight_norm = face_weight / (total_weight + 1e-8)
        
        # ===== APPLY WEIGHTS =====
        weighted_speech = speech_embeddings * speech_weight_norm
        weighted_face = face_embeddings * face_weight_norm
        
        # ===== FUSE EMBEDDINGS =====
        # Concatenate weighted embeddings
        fused = torch.cat([weighted_speech, weighted_face], dim=1)
        
        # Process through fusion layers
        fused_features = self.fusion(fused)
        
        # ===== CLASSIFY =====
        logits = self.classifier(fused_features)
        
        # Return both logits and weights for analysis
        return {
            'logits': logits,
            'speech_weight': speech_weight_norm.squeeze(-1),
            'face_weight': face_weight_norm.squeeze(-1),
            'fused_embeddings': fused_features
        }
