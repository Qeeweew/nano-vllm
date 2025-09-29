import torch
from torch import nn
import os

# -----------------------------------------------------------------------------
# 1. Compile and import the custom kernel
#    Make sure you have re-compiled the kernel with the new RMSNorm op
# -----------------------------------------------------------------------------
try:
    import nanovllm_kernels
    print("Successfully imported custom kernel 'nanovllm_kernels'.")
except ImportError:
    print("ERROR: Failed to import 'nanovllm_kernels'.")
    print("Please ensure you have compiled the C++ extension using 'python setup.py install'.")
    exit()

# -----------------------------------------------------------------------------
# 2. PyTorch reference implementation (from your prompt)
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

# -----------------------------------------------------------------------------
# 3. Core test function
# -----------------------------------------------------------------------------
def run_test(dtype, device):
    print(f"\n--- Running RMSNorm test for dtype={dtype} on device={device} ---")

    # --- Parameter definition ---
    num_tokens = 256
    hidden_size = 4096
    eps = 1e-5 # Use the same eps for both implementations

    # --- Generate input data ---
    # Input 'x' is a 2D tensor
    x_in = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device).contiguous()
    
    print(f"Input shape: {x_in.shape}, dtype={x_in.dtype}")

    # --- 1. Run PyTorch reference implementation ---
    print("\nRunning PyTorch reference implementation...")
    ref_norm_layer = RMSNorm(hidden_size, eps=eps).to(device=device, dtype=dtype)
    # Use the same weight for both implementations for a fair comparison
    weight = ref_norm_layer.weight
    
    ref_output = ref_norm_layer(x_in)
    
    # --- 2. Run custom C++ kernel ---
    print("Running custom C++ kernel...")
    custom_output = nanovllm_kernels.run_rmsnorm(
        x_in,
        weight,
        eps
    )

    # --- 3. Compare results ---
    print("\nComparing outputs...")

    # Set reasonable tolerances for float16 and bfloat16
    atol = 1e-2 if dtype == torch.float16 else 1e-1
    rtol = 1e-3 if dtype == torch.float16 else 1e-2

    try:
        torch.testing.assert_close(ref_output, custom_output, atol=atol, rtol=rtol)
        print("✅ RMSNorm outputs match!")
    except AssertionError as e:
        print("❌ RMSNorm outputs DO NOT match!")
        print(e)
        
    print("-" * 60)


# -----------------------------------------------------------------------------
# 4. Main execution block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if torch.npu.is_available():
        device = "npu:0"
        print(f"Found NPU device: {torch.npu.get_device_name(0)}")
        
        # Test the two supported data types
        run_test(dtype=torch.float16, device=device)
        run_test(dtype=torch.bfloat16, device=device)
        
    else:
        print("WARNING: NPU device not found.")
        print("Cannot test the custom C++ NPU RMSNorm kernel.")