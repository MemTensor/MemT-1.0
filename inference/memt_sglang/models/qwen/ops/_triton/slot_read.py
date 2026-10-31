"""Public Nano reader math with query-only slot indirection.

Keep the original read.py unchanged for public packed()/commit execution.
"""

import triton
import triton.language as tl

BK_READ = 64
BV_READ = 64


def _configs():
    import os

    if os.environ.get("MEMT_NO_AUTOTUNE") == "1":
        return [triton.Config({}, num_warps=4, num_stages=2)]
    return [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ]


@triton.autotune(configs=_configs(), key=["H", "K", "V", "BT"])
@triton.jit
def _slot_read_fwd_kernel(
    q,  # [1, L, H, K]
    snaps,  # [N_block, H, K, V] fp32
    o,
    chunk_tok0,
    chunk_len,
    chunk_block_ids,
    state_slots,
    capacity,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Accumulate each token tile against its assigned snapshot."""
    i_c, i_h, i_v = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2)
    i_block = tl.load(chunk_block_ids + i_c).to(tl.int64)
    i_block = tl.load(state_slots + i_block).to(tl.int64)
    tok0 = tl.load(chunk_tok0 + i_c).to(tl.int64)
    clen = tl.load(chunk_len + i_c)

    o_t = tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    row = tok0 + o_t
    m_t, m_v = o_t < clen, o_v < V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_q = tl.load(
            q + (row[:, None] * H + i_h) * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        )
        b_s = tl.load(
            snaps + (i_block * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :],
            mask=m_k[:, None] & m_v[None, :] & (i_block >= 0) & (i_block < capacity),
            other=0.0,
        )
        b_o += tl.dot(b_q, b_s.to(b_q.dtype))

    b_o *= scale
    tl.store(
        o + (row[:, None] * H + i_h) * V + o_v[None, :],
        b_o.to(o.dtype.element_ty),
        mask=m_t[:, None] & m_v[None, :],
    )
