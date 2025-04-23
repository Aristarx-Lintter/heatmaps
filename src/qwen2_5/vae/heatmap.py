import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel


def fix_nan(x):
    is_bad = torch.isnan(x) | torch.isinf(x)
    if is_bad.any():
        return torch.where(
            is_bad,
            torch.tensor(1e-4, dtype=x.dtype, device=x.device),
            x
        )
    return x


class PostMergerFiLMInjector(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)

        intermediate_dim = hidden_dim // 4 if intermediate_dim is None else intermediate_dim
        self.heatmap_proj = nn.Sequential(
            nn.Linear(1, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 2 * hidden_dim)
        )

        # nn.init.zeros_(self.heatmap_proj[-1].bias)
        # nn.init.zeros_(self.heatmap_proj[-1].weight)

    def forward(self, hidden_states, heatmap_values):
        """
        hidden_states: (N, D)
        heatmap_values: (N, 1)
        """
        normed_hidden_states = self.norm(hidden_states)  # (N, D)
        gamma_beta = self.heatmap_proj(heatmap_values)  # (N, 2*D)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        modulated_states = (1 + gamma) * normed_hidden_states + beta
        output_states = hidden_states + modulated_states

        return output_states


class PostMergerHeatmapInjector(nn.Module):
    def __init__(self, hidden_dim, latent_dim=None):
        super().__init__()
        latent_dim = latent_dim or hidden_dim // 4
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.heatmap_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, hidden_states, heatmap_values):
        """
        hidden_states: (N, D)
        heatmap_values: (N, 1)
        """
        z = self.to_latent(hidden_states)  # (N, latent)
        h = self.heatmap_proj(heatmap_values)  # (N, latent)
        z_combined = z + h  # (N, latent)
        return self.from_latent(z_combined)  # (N, D)


class Qwen2_5_VisionTransformerWithHeatmap(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.post_merger_injector = PostMergerHeatmapInjector(config.out_hidden_size, config.latent_dim)
        self.post_merger_injector = PostMergerFiLMInjector(config.out_hidden_size, config.latent_dim)

    def forward(self, hidden_states: Tensor, grid_thw: Tensor, heatmap_flat: Tensor = None) -> Tensor:
        hidden_states = super().forward(hidden_states, grid_thw)  # (N, D)
        if heatmap_flat is not None:
            hidden_states = self.post_merger_injector(hidden_states, heatmap_flat)  # (N, D)
            is_bad = torch.isnan(hidden_states) | torch.isinf(hidden_states)
            if is_bad.any():
                hidden_states = torch.where(
                    is_bad,
                    torch.tensor(1e-4, dtype=hidden_states.dtype, device=hidden_states.device),
                    hidden_states
                )
                print("NaN detected after post-merger injection")
        return hidden_states



class Qwen2_5_VisionTransformerWithHeatmap(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.post_merger_injector = PostMergerHeatmapInjector(config.out_hidden_size, config.latent_dim)
        self.post_merger_injector = PostMergerFiLMInjector(config.out_hidden_size, config.latent_dim)

    def forward(self, hidden_states: Tensor, grid_thw: Tensor, heatmap_flat: Tensor = None) -> Tensor:
        hidden_states = super().forward(hidden_states, grid_thw)  # (N, D)
        if heatmap_flat is not None:
            hidden_states = self.post_merger_injector(hidden_states, heatmap_flat)  # (N, D)
            is_bad = torch.isnan(hidden_states) | torch.isinf(hidden_states)
            if is_bad.any():
                hidden_states = torch.where(
                    is_bad,
                    torch.tensor(1e-4, dtype=hidden_states.dtype, device=hidden_states.device),
                    hidden_states
                )
                print("NaN detected after post-merger injection")
        return hidden_states