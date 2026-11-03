"""Gather validated published slots directly from FP32 into a BF16 reader buffer."""

import torch


def gather_bf16(pool, slots):
    if not (pool.is_cuda and pool.dtype == torch.float32
            and pool.is_contiguous() and slots.is_contiguous()):
        return pool.index_select(0, slots).to(torch.bfloat16)
    from ._triton.slot_read import gather_cuda

    return gather_cuda(pool, slots)
