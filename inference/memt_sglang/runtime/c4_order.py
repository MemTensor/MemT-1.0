"""Optional logical ordering of native Flash top-k output."""

from functools import wraps
from importlib import import_module

from .features import enabled


def install(model, config=None):
    if not enabled("C4_CANONICAL_ORDER") or model != "memt-flash":
        return
    from memt_sglang.models.deepseek.ops.c4_order import canonicalize_
    native = import_module("sglang.jit_kernel.dsv4.topk")
    indexer = import_module("sglang.srt.layers.attention.dsv4.indexer")
    install_wrappers(indexer, native, canonicalize=canonicalize_)


def install_wrappers(indexer, native, *, canonicalize):
    import torch

    if getattr(indexer, "_memt_c4_logical_order", None) is not None:
        return
    original_v1, original_v2 = native.topk_transform_512, native.topk_transform_512_v2

    @wraps(original_v1)
    def v1(scores, seq_lens, page_tables, out_page_indices, page_size,
           out_raw_indices=None):
        if scores.shape[0] == 0:
            return None
        raw = torch.empty_like(out_page_indices) if out_raw_indices is None else out_raw_indices
        result = original_v1(scores, seq_lens, page_tables, out_page_indices, page_size, raw)
        canonicalize(out_page_indices, raw)
        return result

    @wraps(original_v2)
    def v2(scores, seq_lens, page_tables, out_page_indices, page_size, metadata,
           out_raw_indices=None):
        if scores.shape[0] == 0:
            return None
        raw = torch.empty_like(out_page_indices) if out_raw_indices is None else out_raw_indices
        result = original_v2(scores, seq_lens, page_tables, out_page_indices, page_size, metadata, raw)
        canonicalize(out_page_indices, raw)
        return result

    indexer.topk_transform_512, indexer.topk_transform_512_v2 = v1, v2
    indexer._memt_c4_logical_order = (v1, v2)
