import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm, Qwen2_5VLMLP

class Qwen2_5VisionHeatmapAttentionHead(nn.Module):
    """
    A custom heatmap attention head for Qwen2.5 Vision Transformer.
    This implements cross-attention where:
    - Keys come from the existing vision model features
    - Queries are generated from a heatmap
    
    The heatmap has one value per patch, so the query projection matrix
    has a special shape: (1, hidden_size) to handle this.
    """
    
    def __init__(self, hidden_size, num_heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Special query projection for heatmap values (1 -> hidden_size)
        # Each heatmap value is a scalar (1D) per patch
        self.q_proj = nn.Linear(1, hidden_size, bias=False)
        
        # Key projection matching the vision transformer's representation
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Value projection matching the vision transformer's representation
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Scaling factor for dot product attention
        self.scale = self.head_dim ** -0.5
        
    def forward(self, 
                hidden_states: torch.Tensor,  # Vision features (seq_len, hidden_size)
                heatmap: torch.Tensor,        # Heatmap values (seq_len, 1)
                position_embeddings=None      # Optional positional embeddings from vision transformer
               ) -> torch.Tensor:
        """
        Compute cross-attention between vision features and heatmap.
        
        Args:
            hidden_states: Vision features from the vision transformer (seq_len, hidden_size)
            heatmap: Heatmap values, one scalar per patch (seq_len, 1)
            position_embeddings: Optional position embeddings from the vision transformer
            
        Returns:
            Attended features (seq_len, hidden_size)
        """
        seq_len = hidden_states.size(0)
        
        # Project keys and values from vision features
        k = self.k_proj(hidden_states)  # (seq_len, hidden_size)
        v = self.v_proj(hidden_states)  # (seq_len, hidden_size)
        
        # Project queries from heatmap values
        q = self.q_proj(heatmap)  # (seq_len, hidden_size)
        
        # Reshape for multi-head attention
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        
        # Apply positional embeddings to queries and keys if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Apply rotary position embeddings (RoPE)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            q, k = q_embed, k_embed
        
        # Transpose for batched matrix multiplication
        q = q.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        k = k.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        v = v.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        
        # Compute scaled dot-product attention
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (num_heads, seq_len, seq_len)
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)  # (num_heads, seq_len, head_dim)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(0, 1)  # (seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(seq_len, self.hidden_size)
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class Qwen2_5VisionHeatmapHead(nn.Module):
    """
    A complete heatmap head module for Qwen2.5 Vision Transformer.
    This combines the cross-attention with normalization and MLP layers,
    similar to the structure of a transformer block.
    """
    
    def __init__(self, config):
        """
        Initialize the heatmap head.
        
        Args:
            config: Configuration object with the following attributes:
                - hidden_size: Hidden dimension size
                - num_attention_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Normalization layers
        self.norm1 = Qwen2RMSNorm(config.hidden_size)
        self.norm2 = Qwen2RMSNorm(config.hidden_size)
        
        # Heatmap attention
        self.attn = Qwen2_5VisionHeatmapAttentionHead(
            hidden_size=config.hidden_size,
            num_heads=1  # We use a single attention head for the heatmap
        )
        
        # MLP layer
        self.mlp = Qwen2_5VLMLP(config)
        
    def forward(self, 
                hidden_states: torch.Tensor,   # Vision features (seq_len, hidden_size)
                heatmap: torch.Tensor,         # Heatmap values (seq_len, 1)
                position_embeddings=None       # Optional positional embeddings
               ) -> torch.Tensor:
        """
        Process vision features with heatmap attention.
        
        Args:
            hidden_states: Vision features from the vision transformer
            heatmap: Heatmap values, one scalar per patch
            position_embeddings: Optional position embeddings
            
        Returns:
            Processed features
        """
        # Apply normalization before attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Apply heatmap attention
        hidden_states = self.attn(
            hidden_states=hidden_states,
            heatmap=heatmap,
            position_embeddings=position_embeddings
        )
        
        # First residual connection
        hidden_states = residual + hidden_states
        
        # Apply normalization before MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        
        # Apply MLP
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states

class Qwen2_5VisionHeatmapHeadIntegration:
    """
    Utility class with methods to help integrate the heatmap head
    with the existing Qwen2.5 Vision Transformer model.
    """
    
    @staticmethod
    def extract_features_with_heatmap(vision_model, pixel_values, heatmap, grid_thw):
        """
        Extract features from the vision model and apply heatmap attention.
        
        Args:
            vision_model: Qwen2_5_VisionTransformerPretrainedModel instance
            pixel_values: Input pixel values
            heatmap: Heatmap tensor matching the spatial dimensions of the features
            grid_thw: Grid dimensions (temporal, height, width)
            
        Returns:
            Processed features after applying heatmap attention
        """
        # Create a heatmap head instance
        config = vision_model.config
        heatmap_head = Qwen2_5VisionHeatmapHead(config)
        
        # Get standard vision features from the vision model
        vision_features = vision_model(pixel_values, grid_thw)
        
        # Get position embeddings from the vision model
        rotary_pos_emb = vision_model.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        # Apply heatmap attention
        processed_features = heatmap_head(
            hidden_states=vision_features,
            heatmap=heatmap,
            position_embeddings=position_embeddings
        )
        
        return processed_features 