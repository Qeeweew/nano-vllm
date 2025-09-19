from dataclasses import dataclass
import torch
from typing import Dict, Tuple, Optional


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

    # ===== 新增：全局 Pinned Memory Buffer 管理 =====
    _pinned_buffers: Dict[Tuple[str, Tuple[int, ...], torch.dtype], torch.Tensor] = None

    def __post_init__(self):
        if self._pinned_buffers is None:
            self._pinned_buffers = {}

    def get_pinned_buffer(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        获取或创建一个 pinned memory buffer，自动复用和扩容。
        """
        key = (name, shape, dtype)

        if key in self._pinned_buffers:
            buf = self._pinned_buffers[key]
            # 检查是否需要扩容（当前 buffer 的第一个维度小于请求的）
            if buf.size(0) < shape[0]:
                # 扩容：按 max(2x, new_size) 增长
                new_size0 = max(shape[0], buf.size(0) * 2)
                new_shape = (new_size0,) + shape[1:]
                new_buf = torch.empty(new_shape, dtype=dtype, pin_memory=True)
                self._pinned_buffers[key] = new_buf
                return new_buf[:shape[0]]
            else:
                # 直接切片复用
                return buf[:shape[0]].view(shape)
        else:
            # 首次创建
            buf = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            self._pinned_buffers[key] = buf
            return buf

    def clear_pinned_buffers(self):
        """清空所有 pinned buffer（释放内存）"""
        self._pinned_buffers.clear()


# 全局上下文实例
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
