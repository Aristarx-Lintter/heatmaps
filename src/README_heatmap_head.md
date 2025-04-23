# Qwen2.5 Vision Transformer Heatmap Attention Head

This module implements a custom heatmap attention mechanism for the Qwen2.5 Vision Transformer model. The heatmap head allows injecting spatial attention weights into the vision model's features through a cross-attention mechanism.

## Overview

The heatmap attention head is essentially a cross-attention layer where:
- **Keys** and **Values** come from the existing vision model features
- **Queries** are generated from a heatmap (where each patch has a single scalar value)

This enables the model to focus on specific spatial regions of the input image based on the provided heatmap.

## Technical Details

### Architecture

The implementation consists of several components:

1. `Qwen2_5VisionHeatmapAttentionHead` - The core cross-attention mechanism
   - Special query projection: Takes 1D heatmap values per patch and projects to hidden_size
   - Standard key and value projections from vision features
   - Applies scaled dot-product attention with optional rotary position embeddings

2. `Qwen2_5VisionHeatmapHead` - Complete transformer-like block
   - Includes normalization layers and MLP
   - Follows the architecture pattern of the original vision transformer blocks
   - Residual connections around both attention and MLP

3. `Qwen2_5VisionHeatmapHeadIntegration` - Helper for integration
   - Static methods to help integrate with the existing model
   - Handles extracting features and applying heatmap attention

### Key Implementation Details

- **Query Projection**: Unlike standard attention where Q, K, V all have the same input dimension, here the query input is just 1D (scalar) per patch. The query projection matrix has shape `(1, hidden_size)` to accommodate this.

- **Rotary Position Embeddings**: The implementation supports rotary position embeddings (RoPE) to maintain positional information, matching the original model's approach.

- **Integration**: The implementation is designed to work with the existing Qwen2.5 Vision Transformer model, using the same configuration and compatible with its position embeddings.

## Usage

### Basic Usage

```python
from transformers import Qwen2_5VLProcessor, Qwen2_5VLForConditionalGeneration
from src.heatmap_head_vit import Qwen2_5VisionHeatmapHeadIntegration
import torch

# Load model and processor
model = Qwen2_5VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B")
processor = Qwen2_5VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B")
vision_model = model.vision_model

# Process image
# ... process your image with the processor ...

# Create a heatmap (one value per patch)
seq_len = ...  # Number of patches
heatmap = torch.zeros(seq_len, 1)
# ... set values in your heatmap ...

# Process features with heatmap attention
processed_features = Qwen2_5VisionHeatmapHeadIntegration.extract_features_with_heatmap(
    vision_model, pixel_values, heatmap, grid_thw
)
```

### Creating Heatmaps

The heatmap should be a tensor of shape `(seq_len, 1)` where `seq_len` is the number of patches in the image. Each value represents the attention weight for that patch.

Example ways to create heatmaps:
- From object detection models
- From segmentation masks
- From saliency maps
- From gaze tracking data
- Programmatically defined (e.g., to focus on certain regions)

## Applications

The heatmap attention head can be useful for several applications:

1. **Guided Visual Reasoning**: Guide the model to focus on specific regions when answering visual questions
2. **Visual Grounding**: Improve referring expression comprehension by highlighting relevant regions
3. **Attention Manipulation**: Explore how changing attention patterns affects model predictions
4. **Multimodal Integration**: Use attention maps from one modality to guide processing in another
5. **Human-in-the-loop**: Allow human feedback through attention guidance

## Extension Ideas

- **Multiple Heatmap Heads**: Use multiple heatmaps for different types of guidance
- **Learnable Heatmaps**: Train a small network to generate heatmaps based on the task
- **Bidirectional Integration**: Feed the processed features back to the language model
- **Gating Mechanism**: Add a gate to control how much the heatmap influences the features 