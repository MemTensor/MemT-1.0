"""Write packed GDN2 events and read each request's fixed snapshot once."""

import torch

from memt_sglang._vendor.fla.ops.gdn2.chunk_intra import chunk_gdn2_fwd_intra
from memt_sglang._vendor.fla.ops.utils.constant import RCP_LN2
from memt_sglang._vendor.fla.ops.utils.cumsum import chunk_local_cumsum
from memt_sglang._vendor.fla.utils import input_guard
from memt_sglang.models.common.constants import WY_TILE_TOKENS
from memt_sglang.models.common.packed import boundaries
from memt_sglang.models.common.plan_cache import writer_plan
from memt_sglang.models.qwen.ops._triton.state import memt_gdn2_fwd_h
from memt_sglang.models.qwen.ops.layout import build_chunk_plan
from memt_sglang.models.qwen.ops.read import memt_gdn2_read_fwd


def packed_plan(lengths, device, *, chunk_tokens):
    samples = boundaries(lengths, device)
    events = [0]
    offset = 0
    for length in lengths:
        events.extend(
            offset + token_offset
            for token_offset in range(chunk_tokens, length, chunk_tokens)
        )
        offset += length
        events.append(offset)
    event_slice = torch.tensor(events, device=device, dtype=torch.int32)
    commit_mask = torch.ones(len(events) - 1, device=device, dtype=torch.bool)
    return build_chunk_plan(
        samples, samples, event_slice=event_slice, commit_mask=commit_mask
    )


@input_guard
def token_memory_packed(
    q,
    k,
    v,
    log_decay,
    write,
    read_states,
    write_states,
    *,
    erase,
    lengths,
    scale,
    chunk_tokens,
):
    plan = writer_plan(packed_plan, lengths, q.device, chunk_tokens=chunk_tokens)
    cumulative_decay = chunk_local_cumsum(
        g=log_decay,
        scale=RCP_LN2,
        chunk_size=WY_TILE_TOKENS,
        cu_seqlens=plan.event_slice,
        chunk_indices=plan.chunk_indices,
    )
    write_transform, value_transform, _, scaled_key, _, _ = chunk_gdn2_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=cumulative_decay,
        b=erase,
        w_gate=write,
        scale=scale,
        cu_seqlens=plan.event_slice,
        chunk_size=WY_TILE_TOKENS,
        chunk_indices=plan.chunk_indices,
        safe_gate=False,
        disable_recompute=False,
    )
    _, final_state = memt_gdn2_fwd_h(
        kg=scaled_key,
        w_wy=write_transform,
        u_wy=value_transform,
        gk=cumulative_decay,
        plan=plan,
        initial_state=write_states,
        output_final_state=True,
    )
    read = memt_gdn2_read_fwd(q, read_states, plan, scale, out_dtype=v.dtype)
    return read, final_state
