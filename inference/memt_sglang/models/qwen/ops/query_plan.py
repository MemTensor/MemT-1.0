"""Immutable reader geometry cached per CUDA stream, without device readback."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from memt_sglang.models.common.constants import WY_TILE_TOKENS


@dataclass(frozen=True, slots=True)
class QueryPlan:
    wy_tile_tokens: int
    num_chunk: int
    chunk_tok0: torch.Tensor
    chunk_len: torch.Tensor
    chunk_block_ids: torch.Tensor


@lru_cache(maxsize=64)
def _tile_values(lengths):
    starts, counts, blocks = [], [], []
    offset = 0
    for sample, length in enumerate(lengths):
        for local in range(0, length, WY_TILE_TOKENS):
            starts.append(offset + local)
            counts.append(min(WY_TILE_TOKENS, length - local))
            blocks.append(sample)
        offset += length
    return tuple(starts), tuple(counts), tuple(blocks)


def _build_plan(lengths, device):
    values = _tile_values(lengths)
    tensors = tuple(torch.tensor(v, device=device, dtype=torch.int32) for v in values)
    return QueryPlan(WY_TILE_TOKENS, len(values[0]), *tensors)


@lru_cache(maxsize=64)
def _device_plan(lengths, stream):
    # Keep the real creation stream alive alongside the read-only buffers.
    # Only tile geometry is cached: no RID, slots, state, or layer parameters.
    # Memory readers execute in eager callbacks, including breakable replay.
    with torch.cuda.stream(stream):
        return _build_plan(lengths, stream.device)


def query_plan(lengths, device):
    """Reuse CUDA geometry across layers/forwards on its creation stream.

    This bounded cache is independent of runtime METADATA_CACHE, which controls
    per-forward token/slot metadata. CPU callers need no CUDA context.
    """
    lengths = tuple(lengths)
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("Query lengths must be positive CPU integers")
    if sum(lengths) >= 2**31:
        raise ValueError("Query lengths must fit int32 reader offsets")
    device = torch.device(device)
    if device.type == "cuda":
        return _device_plan(lengths, torch.cuda.current_stream(device))
    return _build_plan(lengths, device)
