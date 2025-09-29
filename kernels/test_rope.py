import torch
import torch.nn as nn
from functools import lru_cache
import os

# -----------------------------------------------------------------------------
# 1. 编译并导入自定义内核
#    确保你已经通过 setup.py 编译并安装了内核
# -----------------------------------------------------------------------------
try:
    import nanovllm_kernels
    print("Successfully imported custom kernel 'nanovllm_kernels'.")
except ImportError:
    print("ERROR: Failed to import 'nanovllm_kernels'.")
    print("Please ensure you have compiled the C++ extension using 'python setup.py install'.")
    exit()

# -----------------------------------------------------------------------------
# 2. 从您的代码中复制纯 Python 的参考实现
# -----------------------------------------------------------------------------
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 确保在进行数学运算时使用 float32
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 在返回前转换回原始 dtype
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
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
        # 使用 float32 计算 inv_freq 和 freqs 以保证精度
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # cos_sin_cache 必须是 float32，以匹配内核的期望
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 从 float32 缓存中获取 cos/sin
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # 将 cos/sin 转换为 query/key 的 dtype 以进行计算
        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        
        query_out = apply_rotary_emb(query, cos, sin)
        key_out = apply_rotary_emb(key, cos, sin)
        return query_out, key_out

@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

# -----------------------------------------------------------------------------
# 3. 核心测试函数
# -----------------------------------------------------------------------------
def run_test(dtype, device):
    print(f"\n--- Running test for dtype={dtype} on device={device} ---")

    # --- 参数定义 ---
    num_tokens = 128 
    num_heads = 32
    kv_num_heads = 4
    head_dim = 128
    max_position = 4096
    base = 10000.0

    # --- 生成输入数据 ---
    # query 和 key 是 3D 张量
    query_in = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device).contiguous()
    key_in = torch.randn(num_tokens, kv_num_heads, head_dim, dtype=dtype, device=device).contiguous()
    
    # positions 是 1D int64 张量
    # 模拟一个非连续的 token position 序列，这在 KV 缓存中很常见
    positions = (torch.rand(num_tokens) * (max_position - 1)).long().to(device)
    
    print(f"Input shapes:")
    print(f"  query:     {query_in.shape}, dtype={query_in.dtype}")
    print(f"  key:       {key_in.shape}, dtype={key_in.dtype}")
    print(f"  positions: {positions.shape}, dtype={positions.dtype}")


    # --- 1. 运行 PyTorch 参考实现 ---
    print("\nRunning PyTorch reference implementation...")
    rotary_emb_ref = get_rope(head_dim, head_dim, max_position, base).to(device)
    ref_query, ref_key = rotary_emb_ref(positions, query_in, key_in)
    
    # --- 2. 运行自定义 C++ 内核 ---
    print("Running custom C++ kernel...")
    # 从参考模型中获取完全相同的 cos_sin_cache
    # 内核期望 float32 类型的 cache
    cos_sin_cache = rotary_emb_ref.cos_sin_cache.to(device) 
    assert cos_sin_cache.dtype == torch.float32, "Cache must be float32 for the kernel"
    
    custom_query, custom_key = nanovllm_kernels.run_rope_custom(
        query_in,          # 第一个参数
        key_in,            # 第二个参数
        positions,         # 第三个参数
        cos_sin_cache      # 第四个参数
    )

    # --- 3. 比较结果 ---
    print("\nComparing outputs...")

    # 为 float16 和 bfloat16 设置合理的容忍度
    atol = 1e-2 if dtype == torch.float16 else 1e-1
    rtol = 1e-3 if dtype == torch.float16 else 1e-2

    # 比较 Query
    try:
        torch.testing.assert_close(ref_query, custom_query, atol=atol, rtol=rtol)
        print("✅ Query outputs match!")
    except AssertionError as e:
        print("❌ Query outputs DO NOT match!")
        print(e)

    # 比较 Key
    try:
        torch.testing.assert_close(ref_key, custom_key, atol=atol, rtol=rtol)
        print("✅ Key outputs match!")
    except AssertionError as e:
        print("❌ Key outputs DO NOT match!")
        print(e)
        
    print("-" * 50)


# -----------------------------------------------------------------------------
# 4. 主执行块
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 根据您的 C++ 代码，这是一个用于华为昇腾 (Ascend) NPU 的内核
    # 因此，我们检查 'npu' 设备是否可用
    if torch.npu.is_available():
        device = "npu:0"
        print(f"Found NPU device: {torch.npu.get_device_name(0)}")
        
        # 测试内核支持的两种数据类型
        run_test(dtype=torch.float16, device=device)
        run_test(dtype=torch.bfloat16, device=device)
        
    else:
        # 如果没有 NPU，则无法运行内核。
        # 我们可以用 CPU 运行参考实现以进行健全性检查，但无法测试内核本身。
        print("WARNING: NPU device not found.")
        print("Cannot test the custom C++ NPU kernel.")
        print("You can still run the reference implementation on CPU for a sanity check.")
        # run_test(dtype=torch.float16, device="cpu") # 这会报错，因为内核不可用