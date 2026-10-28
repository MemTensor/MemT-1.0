"""Launch the tiled snapshot reader for packed Qwen memory queries."""

import torch
import triton

from memt_sglang.models.qwen.ops._triton.read import BK_READ, BV_READ, _read_fwd_kernel
from memt_sglang.models.qwen.ops.layout import ChunkPlan
from memt_sglang.models.qwen.ops.query_plan import QueryPlan


def memt_gdn2_read_fwd(
    q: torch.Tensor,
    snaps: torch.Tensor,
    plan: ChunkPlan | QueryPlan,
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Read each token from its read-block snapshot."""
    _, length, num_heads, key_dim = q.shape
    value_dim = snaps.shape[-1]
    output = torch.empty(
        1, length, num_heads, value_dim, device=q.device, dtype=out_dtype
    )
    grid = (plan.num_chunk, num_heads, triton.cdiv(value_dim, BV_READ))
    _read_fwd_kernel[grid](
        q=q,
        snaps=snaps,
        o=output,
        chunk_tok0=plan.chunk_tok0,
        chunk_len=plan.chunk_len,
        chunk_block_ids=plan.chunk_block_ids,
        scale=scale,
        H=num_heads,
        K=key_dim,
        V=value_dim,
        BT=plan.wy_tile_tokens,
        BK=BK_READ,
        BV=BV_READ,
    )
    return output
