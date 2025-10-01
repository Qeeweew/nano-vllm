import torch
from torch import nn
import torch_npu
from nanovllm_kernels import run_rmsnorm, run_add_rmsnorm


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
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return run_add_rmsnorm(x, residual, self.weight, self.eps)
        return run_rmsnorm(x, self.weight, self.eps)