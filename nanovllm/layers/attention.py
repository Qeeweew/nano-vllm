import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    使用PyTorch将key和value张量存储到分页的KV缓存中。
    这是一个用于替代Triton内核的实现。

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) 每个token对应的线性插槽索引。
    """
    block_size = k_cache.shape[1]

    # 找到有效的插槽（-1表示padding，需要忽略）
    valid_mask = slot_mapping != -1
    if not torch.any(valid_mask):
        return

    valid_slots = slot_mapping[valid_mask]
    # 只选择与有效插槽对应的键/值
    valid_key = key[valid_mask]
    valid_value = value[valid_mask]

    # 将线性插槽索引转换为(块索引, 块内偏移)
    block_indices = torch.div(valid_slots, block_size, rounding_mode='floor')
    block_offsets = valid_slots % block_size

    # 使用高级索引将键和值分散存储到缓存中
    k_cache[block_indices, block_offsets] = valid_key
    v_cache[block_indices, block_offsets] = valid_value


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 如果查询头的数量与键/值头的数量不同，则启用GQA
        self.k_cache = self.v_cache = torch.tensor([])
        if self.num_heads != self.num_kv_heads:
            assert self.num_heads % self.num_kv_heads == 0, "对于GQA，num_heads必须能被num_kv_heads整除"
            self.enable_gqa = True
        else:
            self.enable_gqa = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q shape: (total_tokens, num_heads, head_dim)
        # k,v shape: (total_tokens, num_kv_heads, head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果KV缓存已初始化，则将新的k, v对存入缓存
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            # Prefill阶段：处理具有可变序列长度的提示
            # 由于序列长度不同，我们对batch中的每个序列进行迭代处理
            batch_size = len(context.cu_seqlens_q) - 1
            outputs = []
            
            kv_source_k = k_cache if context.block_tables is not None else k
            kv_source_v = v_cache if context.block_tables is not None else v

            for i in range(batch_size):
                # 切片获取当前序列的query
                q_start, q_end = context.cu_seqlens_q[i], context.cu_seqlens_q[i+1]
                q_i = q[q_start:q_end]
                
                # 获取当前序列的key和value
                k_start, k_end = context.cu_seqlens_k[i], context.cu_seqlens_k[i+1]
                seqlen_k_i = k_end - k_start

                if context.block_tables is not None: # 对于前缀缓存，从分页缓存中收集KV
                    block_table_i = context.block_tables[i]
                    valid_blocks = block_table_i[block_table_i != -1]
                    k_i = kv_source_k[valid_blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen_k_i]
                    v_i = kv_source_v[valid_blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen_k_i]
                else: # KV来自打包的输入张量（无前缀）
                    k_i = kv_source_k[k_start:k_end]
                    v_i = kv_source_v[k_start:k_end]
                    
                # 准备张量以调用scaled_dot_product_attention，其期望shape为(N, H, L, E)
                q_i = q_i.unsqueeze(0).transpose(1, 2)
                k_i = k_i.unsqueeze(0).transpose(1, 2)
                v_i = v_i.unsqueeze(0).transpose(1, 2)
                
                seqlen_q, seqlen_k = q_i.shape[-2], k_i.shape[-2]
                
                causal_mask = None
                if seqlen_q > 0:
                    # 创建一个右下角对齐的因果掩码，以匹配flash-attention的行为
                    q_indices = torch.arange(seqlen_q, device=q.device).view(-1, 1)
                    k_indices = torch.arange(seqlen_k, device=q.device).view(1, -1)
                    causal_mask = (q_indices >= (k_indices - (seqlen_k - seqlen_q))).to(torch.bool)

                o_i = F.scaled_dot_product_attention(
                    q_i, k_i, v_i,
                    attn_mask=causal_mask,
                    scale=self.scale,
                    is_causal=False, # 我们提供了自己的显式掩码
                    enable_gqa=self.enable_gqa
                )
                
                outputs.append(o_i.transpose(1, 2).squeeze(0))

            o = torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)

        else:    # Decode阶段：为每个序列生成一个token
            # q shape: (batch_size, num_heads, head_dim)
            batch_size = q.shape[0]

            # 准备query用于批处理：(B, H, 1, D)
            q = q.unsqueeze(2)

            # 从分页缓存中收集K和V，并填充到最大长度以进行单次批处理调用
            max_seqlen = torch.max(context.context_lens).item()
            k_padded = torch.zeros(batch_size, max_seqlen, self.num_kv_heads, self.head_dim, dtype=q.dtype, device=q.device)
            v_padded = torch.zeros(batch_size, max_seqlen, self.num_kv_heads, self.head_dim, dtype=q.dtype, device=q.device)
            
            for i in range(batch_size):
                seqlen_k_i = context.context_lens[i].item()
                block_table_i = context.block_tables[i]
                valid_blocks = block_table_i[block_table_i != -1]
                
                k_i = k_cache[valid_blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen_k_i]
                v_i = v_cache[valid_blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen_k_i]
                
                k_padded[i, :seqlen_k_i] = k_i
                v_padded[i, :seqlen_k_i] = v_i

            # 为sdpa格式进行转置：(B, H, L, D)
            k_padded = k_padded.transpose(1, 2)
            v_padded = v_padded.transpose(1, 2)
            
            # 创建注意力掩码以忽略k/v中的填充部分
            # 掩码应可广播至(B, H, 1, max_seqlen)
            attn_mask = (torch.arange(max_seqlen, device=q.device)[None, :] < context.context_lens[:, None]).to(torch.bool)

            # --- FIX STARTS HERE ---
            # 原始掩码形状为 (B, S)，需要调整为 (B, 1, 1, S) 以正确广播
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            # --- FIX ENDS HERE ---

            o = F.scaled_dot_product_attention(
                q, k_padded, v_padded,
                attn_mask=attn_mask,
                scale=self.scale,
                is_causal=False, # Decode时q长度为1，非因果
                enable_gqa=self.enable_gqa
            ) # 输出 shape: (B, num_heads, 1, head_dim)
            
            # 重塑输出为 (batch_size, num_heads, head_dim)
            o = o.squeeze(2)
        
        return o
