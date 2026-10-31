"""Batch independent inference-session commits for the WY memory writer.

DeepSeekMemoryBlock.packed calls this from the serving runtime. Sample
offsets isolate each session's read and write state; this is a forward-only path.
"""

from dataclasses import replace

import torch

from memt_sglang._vendor.fla.ops.utils.constant import RCP_LN2
from memt_sglang._vendor.fla.ops.utils.cumsum import chunk_local_cumsum
from memt_sglang.models.common.constants import WY_TILE_TOKENS
from memt_sglang.models.common.plan_cache import writer_plan
from memt_sglang.models.deepseek.ops._triton.intra import write_only_intra
from memt_sglang.models.deepseek.ops._triton.state import memt_gdn2_fwd_h
from memt_sglang.models.deepseek.ops.layout import single_commit_plan
from memt_sglang.models.deepseek.ops.read import snapshot_read


def packed_plan(lengths, device, *, chunk_tokens):
    plans = [
        single_commit_plan(length, WY_TILE_TOKENS, device, chunk_tokens=chunk_tokens)
        for length in lengths
    ]
    events = []
    indices = []
    token_starts = []
    token_lengths = []
    blocks = []
    starts = []
    commits = []
    sample_offsets = [0]
    token, event, chunks = 0, 0, 0
    for sample_index, sample_plan in enumerate(plans):
        events.append(sample_plan.event_slice[:-1] + token)
        indices.append(
            sample_plan.chunk_indices + torch.tensor([event, 0], device=device)
        )
        token_starts.append(sample_plan.chunk_tok0 + token)
        token_lengths.append(sample_plan.chunk_len)
        blocks.append(sample_plan.chunk_block_ids + sample_index)
        starts.append(sample_plan.chunk_is_block_start)
        commits.append(sample_plan.chunk_commits)
        token += sample_plan.total_tokens
        event += sample_plan.num_event
        chunks += sample_plan.num_chunk
        sample_offsets.append(chunks)
    return replace(
        plans[0],
        num_sample=len(plans),
        num_block=len(plans),
        num_event=event,
        num_chunk=chunks,
        total_tokens=token,
        event_slice=torch.cat(
            [*events, torch.tensor([token], device=device, dtype=torch.int32)]
        ),
        chunk_indices=torch.cat(indices).int(),
        sample_chunk_offsets=torch.tensor(
            sample_offsets, device=device, dtype=torch.int32
        ),
        chunk_tok0=torch.cat(token_starts),
        chunk_len=torch.cat(token_lengths),
        chunk_block_ids=torch.cat(blocks),
        chunk_is_block_start=torch.cat(starts),
        chunk_commits=torch.cat(commits),
    )


def token_memory_packed(
    q,
    k,
    v,
    log_decay,
    write,
    read_states,
    write_states,
    *,
    lengths,
    scale,
    chunk_tokens,
):
    plan = writer_plan(packed_plan, lengths, q.device, chunk_tokens=chunk_tokens)
    k, v, log_decay, write = [
        tensor.contiguous() for tensor in (k, v, log_decay, write)
    ]
    cumulative_decay = chunk_local_cumsum(
        g=log_decay,
        scale=RCP_LN2,
        chunk_size=WY_TILE_TOKENS,
        cu_seqlens=plan.event_slice,
        chunk_indices=plan.chunk_indices,
    )
    write_transform, value_transform, scaled_key, _ = write_only_intra(
        k, v, cumulative_decay, None, write, plan
    )
    _, final = memt_gdn2_fwd_h(
        kg=scaled_key,
        w_wy=write_transform,
        u_wy=value_transform,
        gk=cumulative_decay,
        plan=plan,
        initial_state=write_states.contiguous(),
        output_final_state=True,
    )
    return snapshot_read(
        q, read_states, lengths, scale, bf16_accumulator_output=True
    ), final
