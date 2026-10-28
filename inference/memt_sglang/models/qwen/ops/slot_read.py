"""Query-only indirection into published FP32 Nano state slots."""

import torch
import triton

from memt_sglang.models.qwen.ops._triton.slot_read import (
    BK_READ,
    BV_READ,
    _slot_read_fwd_kernel,
)


def slot_query_supported(block, hidden, state_pool, slots, lengths):
    if (
        hidden.ndim != 2
        or hidden.shape[1] != block.hidden_size
        or not hidden.is_cuda
        or hidden.dtype != torch.bfloat16
        or not lengths
        or any(type(n) is not int or n < 1 for n in lengths)
        or sum(lengths) != len(hidden)
        or sum(lengths) >= 2**31
    ):
        return False
    if (
        state_pool.ndim != 4
        or state_pool.shape[0] < 1
        or tuple(state_pool.shape[1:])
        != (block.num_heads, block.head_dim, block.head_dim)
        or state_pool.dtype != torch.float32
        or not state_pool.is_contiguous()
        or state_pool.device != hidden.device
        or slots.ndim != 1
        or slots.numel() != len(lengths)
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
        or slots.device != hidden.device
    ):
        return False

    return True

def slot_snapshot_read(q, snaps, plan, scale, out_dtype, *, state_slots):
    """Read runtime-validated slots using the public BF16 tile/reduction order."""
    _, length, heads, key_dim = q.shape
    value_dim = snaps.shape[-1]
    output = torch.empty(
        (1, length, heads, value_dim), device=q.device, dtype=out_dtype
    )
    grid = (plan.num_chunk, heads, triton.cdiv(value_dim, BV_READ))
    _slot_read_fwd_kernel[grid](
        q=q,
        snaps=snaps,
        o=output,
        chunk_tok0=plan.chunk_tok0,
        chunk_len=plan.chunk_len,
        chunk_block_ids=plan.chunk_block_ids,
        state_slots=state_slots,
        capacity=snaps.shape[0],
        scale=scale,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=plan.wy_tile_tokens,
        BK=BK_READ,
        BV=BV_READ,
    )
    return output
