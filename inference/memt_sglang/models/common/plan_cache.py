"""Optional stream-local immutable geometry cache for eager commit writers."""

from functools import lru_cache

import torch

from memt_sglang.runtime.features import enabled


@lru_cache(maxsize=64)
def _cached_plan(factory, lengths, chunk_tokens, stream):
    with torch.cuda.stream(stream):
        return factory(lengths, stream.device, chunk_tokens=chunk_tokens)


def writer_plan(factory, lengths, device, *, chunk_tokens):
    if not enabled("COMMIT_PLAN_CACHE"):
        return factory(lengths, device, chunk_tokens=chunk_tokens)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Commit plan cache requires eager writer execution")
    # Geometry contains no session IDs, state pointers, or mutable workspaces.
    # A stream key preserves initialization/read ordering without cross-stream
    # synchronization. Unlike query graph constants, these tensors are not
    # captured; normal allocator stream lifetime rules handle LRU eviction.
    return _cached_plan(
        factory, tuple(lengths), chunk_tokens, torch.cuda.current_stream(device)
    )
