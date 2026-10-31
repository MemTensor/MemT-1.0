"""DeepSeek tensor projections and preserved compilation boundaries."""

from functools import cache, lru_cache

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from memt_sglang.models.deepseek.ops.normalization import l2norm


@lru_cache(maxsize=64)
def _epsilon_buffer(values, dtype, stream):
    # Keep immutable constants alive and confined to their initialization stream.
    # In particular, never round host epsilon values through BF16 model weights.
    with torch.cuda.stream(stream):
        return torch.tensor(values, dtype=dtype, device=stream.device)


def grouped_epsilon(values, latent):
    """Materialize host epsilon once, outside the compiled projection graph."""
    dtype = torch.float64 if latent.dtype == torch.float64 else torch.float32
    return _epsilon_buffer(tuple(values), dtype, torch.cuda.current_stream(latent.device))


def _latent_norm(inputs, weight, eps):
    value = (
        inputs.float() if inputs.dtype in (torch.bfloat16, torch.float16) else inputs
    )
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * weight.to(value.dtype)
    ).to(inputs.dtype)


def expand_grouped_qkv(
    q, k, v, q_norm, q_up, q_eps, k_up, v_up, k_norm, v_norm, k_eps, v_eps
):
    """Host low-rank Q and independently normalized K/V latent groups."""
    q = F.linear(_latent_norm(q, q_norm, q_eps), q_up)

    def expand(latent, weight, gain, eps):
        groups, width, rank = weight.shape
        inputs = latent.reshape(-1, groups, rank)
        # Preserve host epsilon precision, including when the block is BF16.
        eps_dtype = torch.float64 if latent.dtype == torch.float64 else torch.float32
        epsilon = torch.as_tensor(eps, dtype=eps_dtype, device=latent.device).view(
            groups, 1
        )
        inputs = _latent_norm(inputs, gain, epsilon)
        output = torch.bmm(inputs.transpose(0, 1), weight.transpose(1, 2))
        return output.transpose(0, 1).reshape(*latent.shape[:-1], groups * width)

    return q, expand(k, k_up, k_norm, k_eps), expand(v, v_up, v_norm, v_eps)


@cache
def resolve_grouped_qkv_expansion():
    return torch.compile(
        expand_grouped_qkv,
        dynamic=True,
        fullgraph=True,
        options={"emulate_precision_casts": True},
    )


def expand_write_gates(
    erase,
    write,
    decay,
    erase_weight,
    write_weight,
    decay_weight,
    decay_bias,
    decay_scale_log,
    heads,
    head_dim,
):
    erase = F.linear(erase, erase_weight, None)
    write = F.linear(write, write_weight, None)
    decay = F.linear(decay, decay_weight, None)
    # Preserve the existing BF16 projection followed by FP32 decay arithmetic.

    gate = F.softplus(decay.float() + decay_bias)
    log_decay = -decay_scale_log.float().exp().view(heads, 1) * gate.view(
        *gate.shape[:-1], heads, -1
    )
    shape = (1, erase.shape[1], heads, head_dim)
    work = erase.float() if erase.dtype in (torch.float16, torch.bfloat16) else erase
    erase_factor = (0.5 * F.logsigmoid(work)).exp().to(erase.dtype)
    return erase_factor.view(shape), write.sigmoid().view(shape), log_decay.view(shape)


def expand_head_projections(
    parts, gate_ups, output_bias, decay_bias, decay_scale_log, heads, head_dim
):
    q, k, v = parts[:3]
    erase, write, log_decay = expand_write_gates(
        *parts[3:6], *gate_ups[:3], decay_bias, decay_scale_log, heads, head_dim
    )
    output = F.linear(parts[6], gate_ups[3], output_bias)
    return q, k, v, output, erase, write, log_decay


@cache
def resolve_head_expansion():
    return torch.compile(
        expand_head_projections,
        dynamic=True,
        fullgraph=True,
        options={"emulate_precision_casts": True},
    )


def prepare_stable_qk(q, k, erase_factor):
    """Normalize raw Q/K once, then multiply K by b or sqrt(b) in consumer dtype."""

    q_work = q.float() if q.dtype in (torch.float16, torch.bfloat16) else q
    k_work = k.float() if k.dtype in (torch.float16, torch.bfloat16) else k
    return l2norm(q_work).to(q.dtype), l2norm(k_work).to(k.dtype) * erase_factor


@cache
def resolve_stable_qk():
    return torch.compile(prepare_stable_qk, dynamic=True, fullgraph=True)


def _gated_rms_norm(
    hidden_states, gate, weight, eps, activation, softcap_enabled, softcap_limit
):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight * hidden_states.to(input_dtype)
    gate_value = ACT2FN[activation](gate.to(torch.float32))
    if softcap_enabled:
        gate_value = softcap_limit * torch.tanh(gate_value / softcap_limit)
    return (hidden_states * gate_value).to(input_dtype)


@cache
def resolve_gated_rms_norm():
    # Keep the eager BF16 rounding points while fusing the wide pointwise chain.
    return torch.compile(
        _gated_rms_norm,
        dynamic=True,
        fullgraph=True,
        options={"emulate_precision_casts": True},
    )
