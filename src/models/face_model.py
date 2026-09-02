"""
Face Emotion Recognition Model
CNN-based facial emotion classification for grayscale images (48x48)
Designed for FER2013 dataset with 4 emotion classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# SIMPLE CNN MODEL


class FaceEmotionModel(nn.Module):
    """
    CNN for facial emotion recognition with embedding extraction support
    
    Input: Grayscale images (batch_size, 1, 48, 48)
    Output: 
      - Logits (batch_size, 4) if return_embeddings=False
      - Embeddings (batch_size, 128) if return_embeddings=True
    
    Architecture:
    - 3 convolutional blocks with batch normalization
    - Max pooling after each conv block
    - 2 fully connected layers
    - Dropout for regularization
    """
    
    def __init__(self, num_emotions=4):
        super(FaceEmotionModel, self).__init__()
        
        # ===== CONVOLUTIONAL LAYERS =====
        self.conv_layers = nn.Sequential(
            # Conv Block 1: (1, 48, 48) -> (32, 24, 24)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2: (32, 24, 24) -> (64, 12, 12)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3: (64, 12, 12) -> (128, 6, 6)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # FEATURE EXTRACTION (Dense layers before classification) 
        # After conv: 128 * 6 * 6 = 4608 features
        self.feature_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),  # Embedding dimension: 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # ===== CLASSIFICATION LAYER =====
        self.classifier = nn.Linear(128, num_emotions)
    
    def forward(self, x, return_embeddings=False):
        """
        Forward pass through the model
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, 1, 48, 48)
            return_embeddings (bool): If True, return embeddings before classification
        
        Returns:
            torch.Tensor: Logits (batch_size, 4) or Embeddings (batch_size, 128)
        """
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Feature extraction
        embeddings = self.feature_layers(x)
        
        # Return embeddings if requested (for feature-level fusion)
        if return_embeddings:
            return embeddings
        
        # Otherwise, return logits (for classification)
        logits = self.classifier(embeddings)
        return logits


# EMOTION MAPPING
EMOTION_NAMES = {
    0: 'angry',
    1: 'happy',
    2: 'neutral',
    3: 'sad'
}

EMOTION_ID_MAP = {v: k for k, v in EMOTION_NAMES.items()}


# TEST / EXAMPLE USAGE
if __name__ == "__main__":
    print("="*70)
    print("FACE EMOTION MODEL - TEST")
    print("="*70 + "\n")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # TEST 1: Simple CNN Model
    print("-" * 70)
    print("TEST 1: Simple CNN Model (FaceEmotionModel)")
    print("-" * 70)
    
    model = FaceEmotionModel(num_emotions=4)
    model = model.to(device)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Create dummy input: batch_size=8, channels=1, height=48, width=48
    print("Input shape: (batch_size=8, channels=1, height=48, width=48)")
    dummy_input = torch.randn(8, 1, 48, 48).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits) sample:\n{output[:2]}\n")
    
    # Get predictions
    probabilities = torch.softmax(output, dim=1)
    predictions = torch.argmax(output, dim=1)
    
    print(f"Probabilities (softmax) sample:\n{probabilities[:2]}")
    print(f"Predicted emotions: {[EMOTION_NAMES[p.item()] for p in predictions]}\n")
    
    
    # TEST 2: ResNet18 Variant 
    print("-" * 70)
    print("TEST 2: ResNet18 Variant (FaceEmotionResNet18) - CORRECTED")
    print("-" * 70)
    print("""
    CRITICAL FIXES APPLIED:
    ✓ Input interpolation: 48x48 → 224x224 (ResNet requirement)
    ✓ First conv layer reinitialized with kaiming_normal_
    ✓ Proper weight initialization for grayscale adaptation
    """)
    
    resnet_model = FaceEmotionResNet18(num_emotions=4, pretrained=True)
    resnet_model = resnet_model.to(device)
    
    print("\nResNet18 Model Architecture (truncated):")
    print(resnet_model)
    
    # Count parameters
    total_params_resnet = sum(p.numel() for p in resnet_model.parameters())
    print(f"\nTotal parameters: {total_params_resnet:,}\n")
    
    # Forward pass
    print("Input shape: (batch_size=8, channels=1, height=48, width=48)")
    print("↓ (internally interpolated to 224x224)")
    with torch.no_grad():
        resnet_output = resnet_model(dummy_input)
    
    print(f"Output shape: {resnet_output.shape}")
    print(f"Output (logits) sample:\n{resnet_output[:2]}\n")
    
    # Get predictions
    resnet_probs = torch.softmax(resnet_output, dim=1)
    resnet_preds = torch.argmax(resnet_output, dim=1)
    
    print(f"Predicted emotions: {[EMOTION_NAMES[p.item()] for p in resnet_preds]}\n")
    
    
    # ===== COMPARISON =====
    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison = f"""
    Model               | Parameters | Input Shape   | Output Shape | Best For
    ────────────────────┼────────────┼───────────────┼──────────────┼──────────────────
    FaceEmotionModel    | {total_params:>9,} | (B, 1, 48, 48) | (B, 4)       | Training from scratch
    FaceEmotionResNet18 | {total_params_resnet:>9,} | (B, 1, 48, 48) | (B, 4)       | Transfer learning
    
    Note: B = batch_size
    ResNet18 has more parameters due to deeper architecture and pretrained features.
    """
    print(comparison)
    
    
    # ===== MULTIMODAL FUSION EXAMPLE =====
    print("="*70)
    print("MULTIMODAL FUSION EXAMPLE")
    print("="*70 + "\n")
    
    print("""
    Example usage in multimodal system:
    
    1. Extract face emotion features:
       face_model = FaceEmotionModel(num_emotions=4)
       face_logits = face_model(face_images)  # (batch_size, 4)
    
    2. Extract speech emotion features (from speech_model.py):
       speech_model = SpeechEmotionModel(num_emotions=4)
       speech_logits = speech_model(mfcc_features)  # (batch_size, 4)
    
    3. Fuse predictions:
       combined_logits = (face_logits + speech_logits) / 2
       # Or use attention mechanism, concatenate, etc.
    
    4. Get final prediction:
       final_probs = torch.softmax(combined_logits, dim=1)
       final_emotion = torch.argmax(final_probs, dim=1)
    """)
    
    print("="*70)
    print("TEST COMPLETE ✓")
    print("="*70)
