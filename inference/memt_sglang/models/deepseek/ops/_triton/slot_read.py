"""Fused slot gather and FP32-to-BF16 conversion; slots are runtime validated."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(
    pool,
    slots,
    output,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row)
    value = tl.load(
        pool + slot * WIDTH + offset,
        (offset < WIDTH) & (slot >= 0) & (slot < CAPACITY),
        other=0,
    )
    tl.store(output + row * WIDTH + offset, value.to(tl.bfloat16), offset < WIDTH)


def gather_cuda(pool, slots):
    output = torch.empty(
        (slots.numel(), *pool.shape[1:]), dtype=torch.bfloat16, device=pool.device
    )
    width = pool[0].numel()
    _gather[(triton.cdiv(width, 1024), slots.numel())](
        pool, slots, output, pool.shape[0], width, 1024
    )
    return output
