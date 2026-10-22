"""Fixed 512-element integer bitonic ordering; no scoring or autotuning."""

import triton
import triton.language as tl


@triton.jit
def canonicalize_kernel(mapped_ptr, raw_ptr):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    raw = tl.load(raw_ptr + row * 512 + col)
    mapped = tl.load(mapped_ptr + row * 512 + col)
    # Valid int32 logical indices < 2**31; invalid positions follow all of them.
    # Original column is a secondary key, retaining pair order for duplicate keys.
    logical_key = tl.where(raw < 0, 2147483648, raw.to(tl.int64))
    keys = (logical_key << 9) | col.to(tl.int64)
    ordered = tl.sort(keys, descending=False)
    permutation = (ordered & 511).to(tl.int32)
    raw_sorted = tl.gather(raw, permutation, axis=0)
    mapped_sorted = tl.gather(mapped, permutation, axis=0)
    tl.store(raw_ptr + row * 512 + col, raw_sorted)
    tl.store(mapped_ptr + row * 512 + col, mapped_sorted)
