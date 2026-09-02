# Multimodal Emotion Recognition: Complete Mathematical Pipeline

## Overview
Complete flow from raw multimodal inputs (speech + face) to emotion classification with entropy-based confidence weighting.

---

## Stage 1: Feature Extraction (Frozen Pretrained Models)

### Speech Embeddings
$$f_s = E_s(x_s) \in \mathbb{R}^{128}$$
- $E_s$: Pretrained speech model encoder
- $x_s$: MFCC features (batch_size, 1, 128, 128)
- $f_s$: 128-dimensional speech embeddings

### Face Embeddings
$$f_f = E_f(x_f) \in \mathbb{R}^{128}$$
- $E_f$: Pretrained face model encoder
- $x_f$: Face image (batch_size, 1, 48, 48)
- $f_f$: 128-dimensional face embeddings

---

## Stage 2: Logit Extraction (For Confidence Computation)

### Speech Logits
$$z_s = C_s(E_s(x_s)) \in \mathbb{R}^{4}$$
- $C_s$: Speech model classifier head
- $z_s$: Raw speech logits (4 emotion classes)

### Face Logits
$$z_f = C_f(E_f(x_f)) \in \mathbb{R}^{4}$$
- $C_f$: Face model classifier head
- $z_f$: Raw face logits (4 emotion classes)

---

## Stage 3: Confidence Computation (Entropy-Based)

### Softmax Probabilities
$$p_s = \text{softmax}(z_s) = \frac{e^{z_s^i}}{\sum_j e^{z_s^j}} \in \mathbb{R}^{4}$$

$$p_f = \text{softmax}(z_f) = \frac{e^{z_f^i}}{\sum_j e^{z_f^j}} \in \mathbb{R}^{4}$$

### Entropy (Uncertainty Measure)
$$H(p_s) = -\sum_{i=1}^{4} p_s^i \log(p_s^i)$$

$$H(p_f) = -\sum_{i=1}^{4} p_f^i \log(p_f^i)$$

- **Low entropy** → confident prediction → high confidence score
- **High entropy** → uncertain prediction → low confidence score

### Normalized Entropy-Based Confidence
$$c_s = 1 - \frac{H(p_s)}{\log(C)}, \quad c_s \in [0, 1]$$

$$c_f = 1 - \frac{H(p_f)}{\log(C)}, \quad c_f \in [0, 1]$$

where $C = 4$ (number of emotion classes)

**Why normalized?** 
- $\log(C) = \log(4) \approx 1.386$ (maximum entropy for 4-class uniform distribution)
- $c = 1 - \frac{H}{H_{\max}}$ ⟹ confidence ∈ [0, 1]

---

## Stage 4: Dynamic Modality Weighting

### Weight Normalization (Softmax-like)
$$w_s = \frac{c_s}{c_s + c_f + \epsilon}$$

$$w_f = \frac{c_f}{c_s + c_f + \epsilon}$$

where $\epsilon = 10^{-8}$ (numerical stability)

**Properties:**
- $w_s + w_f = 1$ (weights form probability distribution)
- $w_s, w_f \in [0, 1]$
- If speech more confident: $w_s > w_f$
- If face more confident: $w_f > w_s$
- If both equally confident: $w_s \approx w_f \approx 0.5$

---

## Stage 5: Weighted Embedding Fusion

### Confidence-Weighted Speech Embeddings
$$\tilde{f}_s = w_s \odot f_s \in \mathbb{R}^{128}$$

### Confidence-Weighted Face Embeddings
$$\tilde{f}_f = w_f \odot f_f \in \mathbb{R}^{128}$$

where $\odot$ denotes element-wise multiplication

**Interpretation:**
- Scale each embedding dimension by its modality's confidence
- High-confidence modality's embeddings have larger magnitude
- Low-confidence modality's embeddings are dampened

### Concatenation
$$f_{\text{concat}} = [\tilde{f}_s; \tilde{f}_f] \in \mathbb{R}^{256}$$

---

## Stage 6: Attention-Based Refinement (Transformer)

### Speech Attention Gate
$$\alpha_s = \text{sigmoid}(\text{MLP}_s(\tilde{f}_s)) \in [0, 1]$$

$$\text{MLP}_s = \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 1) \to \text{Sigmoid}$$

### Face Attention Gate
$$\alpha_f = \text{sigmoid}(\text{MLP}_f(\tilde{f}_f)) \in [0, 1]$$

$$\text{MLP}_f = \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 1) \to \text{Sigmoid}$$

**Why two stages of weighting?**
1. **Confidence weights** ($w_s, w_f$): Fixed global importance (from model certainty)
2. **Attention gates** ($\alpha_s, \alpha_f$): Learned local refinement (learned during fusion training)

### Fusion Layers (Dense Blocks)
$$f'_1 = \text{ReLU}(\text{Linear}(256 \to 256)(f_{\text{concat}}))$$

$$f'_2 = \text{Dropout}(f'_1, p=0.3)$$

$$f'_3 = \text{ReLU}(\text{Linear}(256 \to 128)(f'_2))$$

$$f' = \text{Dropout}(f'_3, p=0.3) \in \mathbb{R}^{128}$$

---

## Stage 7: Final Classification

### Emotion Logits
$$z_{\text{fusion}} = W_c f' + b_c \in \mathbb{R}^{4}$$

- $W_c \in \mathbb{R}^{4 \times 128}$: Classifier weight matrix
- $b_c \in \mathbb{R}^{4}$: Classifier bias

### Final Prediction
$$\hat{y} = \text{softmax}(z_{\text{fusion}}) \in \mathbb{R}^{4}$$

$$\hat{y}_i = \frac{e^{z_{\text{fusion}}^i}}{\sum_{j=1}^{4} e^{z_{\text{fusion}}^j}}$$

---

## Summary: Complete Forward Pass

```
Raw Inputs (Speech + Face)
        ↓
Speech Embeddings: f_s = E_s(x_s)     Face Embeddings: f_f = E_f(x_f)
        ↓                                    ↓
Speech Logits: z_s = C_s(E_s(x_s))   Face Logits: z_f = C_f(E_f(x_f))
        ↓                                    ↓
Speech Confidence: c_s = 1 - H(p_s)/log(4)  Face Confidence: c_f = 1 - H(p_f)/log(4)
        ↓                                    ↓
Modality Weights: w_s = c_s/(c_s + c_f)    w_f = c_f/(c_s + c_f)
        ↓                                    ↓
Weighted Embeddings: f̃_s = w_s ⊙ f_s      f̃_f = w_f ⊙ f_f
        ↓                                    ↓
        └─────────── Concatenation ────────┘
                        ↓
                  f = [f̃_s; f̃_f]
                        ↓
                Attention Refinement
              α_s = sigmoid(MLP_s(f̃_s))
              α_f = sigmoid(MLP_f(f̃_f))
                        ↓
                 Fusion Dense Layers
                   f' = Dense(f)
                        ↓
                  Final Classification
              ŷ = softmax(W_c f' + b_c)
                        ↓
                  Emotion Prediction
```

---

## Training Details

### Loss Function
$$\mathcal{L} = \text{CrossEntropyLoss}(\hat{y}, y_{\text{true}})$$

$$\mathcal{L} = -\sum_{i=1}^{4} y_{\text{true}}^i \log(\hat{y}^i)$$

### Gradient Flow
- Backpropagation through fusion module only
- Speech and face models remain **frozen** (requires_grad=False)
- Only fusion parameters updated: $\theta_{\text{fusion}} = \{W_c, b_c, \text{MLP weights}\}$

---

## Key Innovations

### 1. **Entropy-Based Confidence (Not Accuracy)**
- Uses **certainty** (entropy) of predictions, not accuracy
- Works with frozen models (no retraining needed)
- Reflects model's own doubt in predictions

### 2. **Two-Stage Weighting Architecture**
- **Confidence weights** ($w_s, w_f$): Global, fixed during training
- **Attention gates** ($\alpha_s, \alpha_f$): Local, learned during training
- Attention refines pre-weighted embeddings without replacing confidence logic

### 3. **Embedding-Space Fusion (Not Prediction-Space)**
- Weights applied to features ($f_s, f_f$), not logits
- Preserves spatial structure and interactions
- Fusion module learns complementary representations

### 4. **Preserved One-to-One Mapping**
- Each utterance maintains:
  - Unique extracted face image (from IEMOCAP video)
  - Unique emotion label from EmoEvaluation file
  - Traceability through utterance ID
  - Complete metadata (time, speaker, session)

---

## Dimensions Summary

| Component | Input Dim | Output Dim | Notes |
|-----------|-----------|-----------|-------|
| Speech Embeddings | - | 128 | From E_s encoder |
| Face Embeddings | - | 128 | From E_f encoder |
| Speech Logits | - | 4 | For confidence only |
| Face Logits | - | 4 | For confidence only |
| Confidence Scores | - | 1 | Scalar [0, 1] |
| Modality Weights | - | 1 | Scalar [0, 1] |
| Weighted Embeddings | 128 | 128 | Scaled by weight |
| Concatenated | - | 256 | [f̃_s; f̃_f] |
| Fused Embeddings | 256 | 128 | After dense layers |
| Final Logits | 128 | 4 | Emotion classes |
| Final Prediction | - | 4 | Softmax probabilities |

---

## Mathematical Verification

**Constraint 1: Weights sum to 1**
$$w_s + w_f = \frac{c_s + c_f}{c_s + c_f} = 1 \quad \checkmark$$

**Constraint 2: Confidence in [0, 1]**
$$c = 1 - \frac{H}{\log(C)}: \quad 0 \leq H \leq \log(C) \implies 0 \leq c \leq 1 \quad \checkmark$$

**Constraint 3: Entropy maximized at uniform**
$$H_{\max} = \log(C) \text{ when } p_i = \frac{1}{C} \, \forall i \quad \checkmark$$

**Constraint 4: Attention weights in [0, 1]**
$$\alpha = \text{sigmoid}(x) \in (0, 1) \quad \checkmark$$

---

## Code Correspondence

| Mathematical Notation | Implementation Location |
|----------------------|--------------------------|
| $c_s, c_f$ | `train_multimodal_fusion.py`: `get_entropy_based_confidence()` |
| $w_s, w_f$ | `train_multimodal_fusion.py`: Weight normalization |
| $\tilde{f}_s, \tilde{f}_f$ | `train_multimodal_fusion.py`: `apply_confidence_weights()` |
| $\alpha_s, \alpha_f$ | `embedding_fusion.py`: `speech_gate`, `face_gate` |
| $f'$ | `embedding_fusion.py`: `fusion` sequential block |
| $\hat{y}$ | `embedding_fusion.py`: `classifier` linear layer |

---

This is the **complete, mathematically rigorous formulation** of the multimodal fusion pipeline with entropy-based confidence weighting.
