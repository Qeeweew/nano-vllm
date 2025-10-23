from dataclasses import dataclass, field
import torch
from typing import Dict, Tuple, Optional, Any


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    moe_tracker: Optional[Any] = None
    
    # --- START MODIFICATION ---
    # This flag will tell get_pinned_buffer whether we are in a graph replay.
    is_graph_captured: bool = False 
    
    # These will hold the pre-allocated buffers during graph replay.
    graph_moe_hidden_buffer: Optional[torch.Tensor] = None
    graph_moe_logits_buffer: Optional[torch.Tensor] = None
    # --- END MODIFICATION ---

    # ===== Global Pinned Memory Buffer Management (for eager mode) =====
    _pinned_buffers: Dict[Tuple[str, Tuple[int, ...], torch.dtype], torch.Tensor] = field(default_factory=dict, init=False)


    def get_pinned_buffer(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Gets a pinned memory buffer.
        - In EAGER mode: Manages a pool of reusable buffers.
        - In CUDA GRAPH mode: Returns a slice of a pre-allocated static buffer.
        """
        # --- START MODIFICATION ---
        if self.is_graph_captured:
            # We are in a graph. Return a slice of the static buffer passed via the context.
            if name == "moe_hidden":
                buffer = self.graph_moe_hidden_buffer
            elif name == "moe_logits":
                buffer = self.graph_moe_logits_buffer
            else:
                raise ValueError(f"Unknown pinned buffer name in graph mode: {name}")

            # The full buffer was pre-allocated to max size. Slice it to the current required size.
            return buffer[:shape[0]]
        # --- END MODIFICATION ---

        # --- Eager mode logic (unchanged) ---
        key = (name, shape, dtype)

        if key in self._pinned_buffers:
            buf = self._pinned_buffers[key]
            if buf.size(0) < shape[0]:
                new_size0 = max(shape[0], buf.size(0) * 2)
                new_shape = (new_size0,) + shape[1:]
                new_buf = torch.empty(new_shape, dtype=dtype, device="cpu", pin_memory=True)
                self._pinned_buffers[key] = new_buf
                return new_buf[:shape[0]]
            else:
                return buf[:shape[0]].view(shape)
        else:
            buf = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            self._pinned_buffers[key] = buf
            return buf

    def clear_pinned_buffers(self):
        """Clears all pinned buffers (releases memory in eager mode)."""
        self._pinned_buffers.clear()


# Global context instance
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, moe_tracker=None, **kwargs):
    global _CONTEXT
    # Pass kwargs to the Context constructor to handle new fields
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, moe_tracker, **kwargs)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()