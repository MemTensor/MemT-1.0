"""Compiler pointwise fusion while retaining eager's FP32 reduction order."""

from functools import cache

import torch


def square_fp32(hidden):
    value = hidden.float()
    return value * value


def finish(hidden, variance, gate, weight, eps, limit):
    dtype = hidden.dtype
    normalized = hidden.float() * torch.rsqrt(variance + eps)
    normalized = weight * normalized.to(dtype)
    value = gate.float()
    # Eager CUDA SiLU divides by (1+exp(-x)); avoid tl.sigmoid's approximate
    # division/exp2 lowering. The division-rounding compiler option is required.
    activated = value / (1.0 + torch.exp(-value))
    activated = limit * torch.tanh(activated / limit)
    return (normalized * activated).to(dtype)


@cache
def resolve_gated_rms_norm():
    options = {
        "emulate_precision_casts": True,
        "eager_numerics.division_rounding": True,
        "use_fast_math": False,
        "triton.cudagraphs": False,
    }
    squared = torch.compile(square_fp32, dynamic=True, fullgraph=True, options=options)
    output = torch.compile(finish, dynamic=True, fullgraph=True, options=options)

    def normalize(hidden, gate, weight, eps, limit):
        variance = squared(hidden).mean(-1, keepdim=True)
        return output(hidden, variance, gate, weight, eps, limit)

    return normalize
