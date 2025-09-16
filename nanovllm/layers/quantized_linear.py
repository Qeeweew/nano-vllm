# FILE: ./nanovllm/layers/quantized_linear.py
import torch
from torch import nn
import nanovllm_ext  # This will be built by setup.py


class QuantizedLinear(nn.Module):
    """
    A quantized linear layer for CPU execution, using a custom FP32 x int8 -> FP32 GEMM kernel.
    The weight is quantized and repacked offline.
    """
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # register_buffer ensures the tensors are moved to the correct device
        # with the model, but they are not considered model parameters.
        self.register_buffer("weight_qs", torch.empty(0, dtype=torch.int8))
        self.register_buffer("weight_d", torch.empty(0, dtype=torch.half))

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, device="cpu"))
        else:
            self.register_parameter("bias", None)
        
        self._quantized = False

    def quantize(self, weight: torch.Tensor):
        """
        Quantizes the float weight tensor and stores it in the packed format.

        Args:
            weight (torch.Tensor): The FP32 weight tensor of shape (output_size, input_size).
        """
        assert weight.shape == (self.output_size, self.input_size)
        assert weight.device.type == 'cpu', "Weight quantization must happen on CPU"

        weight_qs, weight_d = nanovllm_ext.quantize_repack_weight(weight.contiguous().to(torch.float32))
        
        # Assign to the buffers
        self.weight_qs = weight_qs
        self.weight_d = weight_d
        
        self._quantized = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._quantized:
            raise RuntimeError("Linear layer has not been quantized. Call .quantize(weight) first.")

        # Store original dtype and shape to restore them later.
        assert x.dtype == torch.float32
        orig_shape = x.shape

        # The GEMM kernel expects a 2D input
        x_reshaped = x.view(-1, self.input_size)
        
        # Call the C++ extension. It will return a float32 tensor.
        y = nanovllm_ext.q8_gemm(x_reshaped, self.weight_qs, self.weight_d)
        
        # Reshape the output to match the input batch dimensions
        y = y.view(*orig_shape[:-1], self.output_size)

        if self.bias is not None:
            y += self.bias.to(torch.float32)
        
        return y

    def extra_repr(self) -> str:
        return (f'input_size={self.input_size}, output_size={self.output_size}, '
                f'bias={self.bias is not None}, quantized={self._quantized}')