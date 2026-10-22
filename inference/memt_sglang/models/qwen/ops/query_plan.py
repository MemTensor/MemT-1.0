"""Reader-only tiles from CPU lengths; no GPU-to-host boundary synchronization."""

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


def query_plan(lengths, device):
    lengths = tuple(lengths)
    values = _tile_values(lengths)
    # Cache only immutable host values. Device buffers belong to this eager
    # invocation's stream, including when it is a breakable graph callback.
    tensors = tuple(torch.tensor(v, device=device, dtype=torch.int32) for v in values)
    return QueryPlan(WY_TILE_TOKENS, len(values[0]), *tensors)
