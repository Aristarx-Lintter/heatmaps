import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor
import torch.nn.functional as F


from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import flash_attn_varlen_func, apply_rotary_pos_emb_flashatt
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm, Qwen2_5_VLMLP, Qwen2_5_VLVisionFlashAttention2
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel


class HeatmapEmbeddingLayer(nn.Module):
    def __init__(self, hidden_state: int):
        super().__init__()
        self.linear1 = nn.Linear(1, hidden_state)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(hidden_state, hidden_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1(x)
        output = self.linear2(self.activation(h1))
        
        return output



class Qwen2_5_VLCrossAttentionFlashAttention2(nn.Module):
    """
    Модифицированный Attention модуль для Cross-Attention с использованием FlashAttention v2.
    Query (Q) берется из `context_features` (например, признаки тепловых карт).
    Key (K) и Value (V) берутся из `hidden_states` (например, визуальные признаки).
    Ротационные эмбеддинги (`position_embeddings`), предназначенные для `hidden_states`,
    применяются к Q и K для сохранения структуры (требует совпадения длин!).
    """
    def __init__(self, hidden_state: int, num_heads: int = 16, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_state // num_heads

        self.q = nn.Linear(hidden_state, hidden_state, bias=bias)
        self.kv = nn.Linear(hidden_state, hidden_state * 2, bias=bias)
        self.proj = nn.Linear(hidden_state, hidden_state)
        
    def forward(
        self,
        hidden_states: torch.Tensor,                # Фичи для K, V, shape: (total_seq_len_kv, dim)
        context_features: torch.Tensor,             # Фичи для Q, shape: (total_seq_len_q, dim_context)
        cu_seqlens: torch.Tensor,                   # Кумулятивные длины для K/V (hidden_states), shape: (batch_size + 1,)
        # cu_seqlens_context: torch.Tensor,           # Кумулятивные длины для Q (context), shape: (batch_size + 1,)
        rotary_pos_emb: Optional[torch.Tensor] = None, # Устаревший способ передачи RoPE (для K/V)
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # (cos, sin) для RoPE (для K/V)
                                                                                # shape: (total_seq_len_kv, rotary_dim) или совместимые
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Тензор для вычисления Key и Value. Форма (total_seq_len_kv, dim).
            context_features: Тензор для вычисления Query. Форма (total_seq_len_q, dim_context).
            cu_seqlens: Кумулятивные длины последовательностей для `hidden_states`. Форма (batch_size + 1,).
            cu_seqlens_context: Кумулятивные длины последовательностей для `context_features`. Форма (batch_size + 1,).
            rotary_pos_emb: Theta значения RoPE (устарело). Используется, если position_embeddings is None.
            position_embeddings: Кортеж (cos, sin) для RoPE. Ожидается, что они рассчитаны для `hidden_states`.

        Returns:
            Тензор выхода attention. Форма (total_seq_len_q, dim).
        """
        seq_length_kv = hidden_states.shape[0]  
        seq_length_q = context_features.shape[0]  

        kv = self.kv(hidden_states).reshape(seq_length_kv, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)
        q = self.q(context_features).reshape(seq_length_q, self.num_heads, self.head_dim)

        if position_embeddings is None:
            if rotary_pos_emb is None:
                 raise ValueError("Provide either position_embeddings or rotary_pos_emb")
            if rotary_pos_emb.shape[0] != seq_length_kv:
                 raise ValueError(f"rotary_pos_emb have length {rotary_pos_emb.shape[0]}, but expected {seq_length_kv}")
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
            if cos.shape[0] != seq_length_kv or sin.shape[0] != seq_length_kv:
                 raise ValueError(f"position_embeddings have {cos.shape[0]}/{sin.shape[0]}, but expected {seq_length_kv}")


        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        max_seqlen_kv = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        q = q.squeeze(0)
        k = k.squeeze(0)
        
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen_kv, max_seqlen_kv,
            causal=False, # Cross-attention не каузальное
        )
        # attn_output shape: (total_q, num_heads, head_dim)
        # print(f"CrossAttn: attn_output shape after flash_attn: {attn_output.shape}")

        # (total_q, num_heads, head_dim) -> (total_q, dim)
        attn_output = attn_output.reshape(seq_length_q, -1)
        attn_output = self.proj(attn_output)
        # print(f"CrossAttn: final attn_output shape after projection: {attn_output.shape}")

        return attn_output



class Qwen2_5_VLVisionBlockHeat(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm0 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm3 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn_self = Qwen2_5_VLVisionFlashAttention2(
            config.hidden_size, num_heads=config.num_heads
        )
        self.attn_cross = Qwen2_5_VLCrossAttentionFlashAttention2(
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_features: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn_self(
            self.norm0(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        
        hidden_states = hidden_states + self.attn_cross(
            self.norm1(hidden_states),
            self.norm2(context_features),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        
        hidden_states = hidden_states + self.mlp(self.norm3(hidden_states))
        return hidden_states


class Qwen2_5_VisionTransformerWithHeatmap(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.post_merger_injector = PostMergerHeatmapInjector(config.out_hidden_size, config.latent_dim)
        # self.post_merger_injector = PostMergerFiLMInjector(config.out_hidden_size, config.latent_dim)
        self.heat_block = Qwen2_5_VLVisionBlockHeat(config)
        
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, heatmap_flat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
            heatmap_flat (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The heatmap of each image in LLM.
        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            if self.gradient_checkpointing and self.training:
                args = [blk.__call__, hidden_states]
                hidden_states = self._gradient_checkpointing_func(*args, cu_seqlens_now, None, position_embeddings)
                if heatmap_flat is not None and layer_num == self.fullatt_block_indexes[-2]:
                    args = [self.heat_block.__call__, hidden_states, heatmap_flat]
                    hidden_states = self._gradient_checkpointing_func(*args, cu_seqlens_now, None, position_embeddings)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
                if heatmap_flat is not None and layer_num == self.fullatt_block_indexes[-2]:
                    injection = {"context_features": heatmap_flat}
                    hidden_states = self.heat_block(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, **injection)


        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states