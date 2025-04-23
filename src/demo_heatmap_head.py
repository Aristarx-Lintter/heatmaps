import torch
import numpy as np
from transformers import Qwen2_5VLProcessor, Qwen2_5VLForConditionalGeneration
from PIL import Image
from src.heatmap_head_vit import Qwen2_5VisionHeatmapHeadIntegration

def generate_sample_heatmap(feature_shape, peak_position=None):
    """
    Generate a sample heatmap with a gaussian peak.
    
    Args:
        feature_shape: Shape of the feature map (seq_len, 1)
        peak_position: Position of the peak in the heatmap. If None, a random position is used.
        
    Returns:
        heatmap: Tensor with shape (seq_len, 1)
    """
    seq_len = feature_shape[0]
    
    # Create a flat heatmap
    heatmap = torch.zeros(seq_len, 1)
    
    # Set a peak at a specific position or random
    if peak_position is None:
        peak_position = np.random.randint(0, seq_len)
    
    # Create a gaussian peak centered at peak_position
    positions = torch.arange(seq_len)
    gaussian = torch.exp(-0.5 * ((positions - peak_position) / 10.0) ** 2)
    heatmap[:, 0] = gaussian
    
    # Normalize heatmap values
    heatmap = heatmap / heatmap.max()
    
    return heatmap

def main():
    # Load model and processor
    model = Qwen2_5VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B")
    processor = Qwen2_5VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B")
    
    # Get the vision model
    vision_model = model.vision_model
    
    # Sample image
    image = Image.open("sample_image.jpg")  # Replace with your image path
    
    # Process image
    vision_inputs = processor.image_processor(image, return_tensors="pt")
    pixel_values = vision_inputs.pixel_values
    
    # Prepare grid_thw parameter
    # For a single image, this would typically be [1, H, W] where H and W are the
    # number of patches in the height and width dimensions
    # Let's assume a 224x224 image with patch size 14
    patch_size = vision_model.patch_size
    grid_h = 224 // patch_size
    grid_w = 224 // patch_size
    grid_thw = torch.tensor([[1, grid_h, grid_w]])
    
    # Process image with vision model to get features
    with torch.no_grad():
        vision_features = vision_model(pixel_values, grid_thw)
    
    # Generate a sample heatmap
    # The heatmap should have one value per patch
    seq_len = vision_features.shape[0]
    heatmap = generate_sample_heatmap((seq_len, 1))
    
    # Process features with heatmap attention
    processed_features = Qwen2_5VisionHeatmapHeadIntegration.extract_features_with_heatmap(
        vision_model, pixel_values, heatmap, grid_thw
    )
    
    print(f"Original features shape: {vision_features.shape}")
    print(f"Processed features shape: {processed_features.shape}")
    
    # Now you can use these processed features for your downstream task
    # For example, you could feed them back into the model's language part
    # or use them for a custom classification task

if __name__ == "__main__":
    main() 