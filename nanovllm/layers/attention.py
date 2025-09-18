import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.utils.context import get_context
import torch_npu

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

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
            o = torch.empty_like(q)
            # 调用高性能的 _npu_paged_attention 算子
            torch_npu._npu_paged_attention(
                q,                      # query: (bs, num_heads, head_dim)
                self.k_cache,           # key_cache: (num_blocks, block_size, num_kv_heads, head_dim)
                self.v_cache,           # value_cache: (num_blocks, block_size, num_kv_heads, head_dim)
                self.num_kv_heads,      # kv_heads (int)
                self.num_heads,         # num_heads (int)
                self.scale,             # scale (float)
                context.block_tables,   # block_tables: (bs, max_blocks_per_seq)
                context.context_lens,   # context_lens: (bs,)，必须在 CPU 上
                o                       # output: (bs, num_heads, head_dim)
            )        
        return o
