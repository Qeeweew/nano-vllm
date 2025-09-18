from functools import lru_cache
import torch
import torch_npu
from torch import nn

class RotaryEmbedding(nn.Module):
    """
    使用 torch_npu.npu_apply_rotary_pos_emb 优化 RoPE 的前向计算。
    合并 cos_cache 和 sin_cache 为 cos_sin_cache。
    """
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        # 1. 计算原始的半维度 cos 和 sin
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos_half = freqs.cos().to(torch.get_default_dtype())
        sin_half = freqs.sin().to(torch.get_default_dtype())

        # 2. 构造全维度 cos 和 sin（复制拼接）
        cos_full = torch.cat((cos_half, cos_half), dim=-1)
        sin_full = torch.cat((sin_half, sin_half), dim=-1)

        # 3. 合并为 cos_sin_cache: [L, D*2]
        cos_sin_cache = torch.cat((cos_full, sin_full), dim=-1)

        # 4. 注册为单个 buffer
        self.register_buffer("cos_sin_cache", cos_sin_cache.unsqueeze_(1).unsqueeze_(1), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用 NPU 融合算子进行高效计算。
        """
        rope = self.cos_sin_cache[positions]
        cos = rope[..., :self.head_size]
        sin = rope[..., self.head_size:]

        query = query.unsqueeze_(1)
        key = key.unsqueeze_(1)

        # query, key 形状: [total_tokens, 1, num_heads, head_dim]
        # cos, sin 形状: [total_tokens, 1, 1, head_dim]
        query_out, key_out = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin)

        # 移除插入的维度
        query_out = query_out.squeeze_(1)
        key_out = key_out.squeeze_(1)

        return query_out, key_out


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
