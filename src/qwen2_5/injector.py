import torch
import torch.nn as nn


class HeatmapGate(nn.Module):
    """
    Небольшой модуль, отвечающий за пер-головное умножение логитов внимания
    на (1 + gamma^head * heatmap).
    """
    def __init__(self, num_heads):
        super().__init__()
        # обучаемый параметр per-head
        self.gamma = nn.Parameter(torch.zeros(num_heads))

    def forward(self, attn_scores, heatmap_bias):
        """
        attn_scores: (batch_size, num_heads, q_len, k_len)
        heatmap_bias: (batch_size, k_len) или (batch_size, 1, 1, k_len)
        """
        b, h, q_len, k_len = attn_scores.shape
        # Приводим heatmap_bias к (B, 1, 1, k_len)
        if heatmap_bias.dim() == 2:
            heatmap_bias = heatmap_bias.unsqueeze(1).unsqueeze(2)
        # gamma -> (1, num_heads, 1, 1)
        gamma_4d = self.gamma.view(1, -1, 1, 1)
        # Инъекция
        out = attn_scores * (1.0 + gamma_4d * heatmap_bias)
        return out
