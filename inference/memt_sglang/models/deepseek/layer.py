"""DeepSeek grouped-KV memory with checkpoint parameter names."""

import torch
from torch import nn

from memt_sglang.models.common.packed import validate_packed_inputs
from memt_sglang.models.deepseek.config import DeepSeekMemoryConfig
from memt_sglang.models.deepseek.ops.compiled_tensor_ops import (
    resolve_gated_rms_norm,
    resolve_grouped_qkv_expansion,
    resolve_head_expansion,
    resolve_stable_qk,
)
from memt_sglang.models.deepseek.ops.packed import token_memory_packed
from memt_sglang.models.deepseek.ops.read import packed_snapshot_read


class _LatentNorm(nn.Module):
    def __init__(self, rank, norm_eps, *, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(rank, device=device, dtype=dtype))
        self.eps = norm_eps


class _GatedRMSNorm(nn.Module):
    def __init__(self, head_dim, norm_eps, softcap, *, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim, device=device, dtype=dtype))
        self.variance_epsilon = norm_eps
        self.softcap = softcap

    def forward(self, value, output_gate):
        head_dim = value.shape[-1]
        output = resolve_gated_rms_norm()(
            value.reshape(-1, head_dim),
            output_gate.reshape(-1, head_dim),
            self.weight,
            self.variance_epsilon,
            activation="silu",
            softcap_enabled=True,
            softcap_limit=self.softcap,
        )
        return output.reshape_as(value)


class DeepSeekMemoryBlock(nn.Module):
    """CUDA BF16 projections and GDN2 updates over an FP32 memory state.

    All tokens read the incoming snapshot. A commit produces a separate final
    state; a query returns no state update. The scheduler controls
    publication and reset of that state.
    """

    def __init__(
        self, config: DeepSeekMemoryConfig, *, device=None, dtype=torch.bfloat16
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.value_dim = config.num_heads * config.head_dim
        self.read_scale = config.head_dim**-0.5
        self.kv_groups = config.kv_groups
        self.chunk_tokens = config.chunk_tokens
        self.gate_rows = (config.gate_rank,) * 4
        self.qkv_rows = (
            config.q_rank,
            config.kv_groups * config.kv_rank,
            config.kv_groups * config.kv_rank,
        )

        factory = {"device": device, "dtype": dtype}
        self.in_proj = nn.Linear(
            config.hidden_size,
            sum(self.qkv_rows) + sum(self.gate_rows),
            bias=False,
            **factory,
        )
        self.q_norm = _LatentNorm(config.q_rank, config.norm_eps, **factory)
        self.q_up = nn.Linear(config.q_rank, self.value_dim, bias=False, **factory)
        group_width = self.value_dim // config.kv_groups
        # These HF keys have no '.weight' suffix, so they are bare Parameters.
        self.k_group_up = nn.Parameter(
            torch.empty(config.kv_groups, group_width, config.kv_rank, **factory)
        )
        self.v_group_up = nn.Parameter(
            torch.empty(config.kv_groups, group_width, config.kv_rank, **factory)
        )
        self.k_group_norm = nn.Parameter(
            torch.ones(config.kv_groups, config.kv_rank, **factory)
        )
        self.v_group_norm = nn.Parameter(
            torch.ones(config.kv_groups, config.kv_rank, **factory)
        )
        self.k_group_norm_eps = (config.norm_eps,) * config.kv_groups
        self.v_group_norm_eps = (config.norm_eps,) * config.kv_groups

        self.b_up = nn.Linear(config.gate_rank, self.value_dim, bias=False, **factory)
        self.w_up = nn.Linear(config.gate_rank, self.value_dim, bias=False, **factory)
        self.f_down = nn.Linear(config.gate_rank, self.value_dim, bias=False, **factory)
        self.g_down = nn.Linear(config.gate_rank, self.value_dim, bias=True, **factory)
        self.decay_scale_log = nn.Parameter(torch.zeros(config.num_heads, **factory))
        self.decay_bias = nn.Parameter(torch.zeros(self.value_dim, **factory))
        self.o_norm = _GatedRMSNorm(
            config.head_dim, config.norm_eps, config.output_softcap, **factory
        )

    def _project(self, hidden_states):
        """Preserve packed projection strides and separate compilation graphs."""

        # Training projects packed [1,T,C] then compiles these two separate
        # expansion graphs. Preserve latent strides and compilation boundaries.
        parts = self.in_proj(hidden_states.unsqueeze(0)).split(
            (*self.qkv_rows, *self.gate_rows), dim=-1
        )
        # Full-head slices still matter: training passes Tensor views here.
        # Bare Parameters specialize differently under dynamic torch.compile
        # and change grouped K/V rounding despite identical values/strides.
        q, k, v = resolve_grouped_qkv_expansion()(
            *parts[:3],
            self.q_norm.weight,
            self.q_up.weight[:],
            self.q_norm.eps,
            self.k_group_up[:],
            self.v_group_up[:],
            self.k_group_norm[:],
            self.v_group_norm[:],
            self.k_group_norm_eps,
            self.v_group_norm_eps,
        )
        outputs = resolve_head_expansion()(
            (q, k, v, *parts[3:]),
            (
                self.b_up.weight[:],
                self.w_up.weight[:],
                self.f_down.weight[:],
                self.g_down.weight[:],
            ),
            self.g_down.bias[:],
            self.decay_bias[:],
            self.decay_scale_log[:],
            self.num_heads,
            self.head_dim,
        )
        return tuple(
            projection_output.reshape(
                1, len(hidden_states), self.num_heads, self.head_dim
            )
            for projection_output in outputs
        )

    @torch.no_grad()
    def packed(self, hidden_states, read_states, *, lengths, commit, write_states):
        validate_packed_inputs(self, hidden_states, read_states, write_states, lengths)

        return self._compute_packed(
            hidden_states,
            read_states,
            lengths=lengths,
            commit=commit,
            write_states=write_states,
        )

    @torch.no_grad()
    def packed_query_slots(self, hidden_states, state_pool, slots, *, lengths):
        from memt_sglang.models.deepseek.ops.slot_read import gather_bf16

        if (hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size or (not hidden_states.is_cuda) or (hidden_states.dtype != torch.bfloat16) or (not lengths) or any((type(length) is not int or length < 1 for length in lengths)) or (sum(lengths) != len(hidden_states))) or (state_pool.shape[1:] != (self.num_heads, self.head_dim, self.head_dim) or state_pool.device != hidden_states.device or slots.numel() != len(lengths)):
            states = state_pool.index_select(0, slots)
            return self.packed(hidden_states, states, lengths=lengths,
                               commit=False, write_states=states)
        read_states = gather_bf16(state_pool, slots)
        return self._compute_packed(
            hidden_states,
            read_states,
            lengths=lengths,
            commit=False,
            write_states=None,
        )

    def _compute_packed(
        self,
        hidden_states,
        read_states,
        *,
        lengths,
        commit,
        write_states,
    ):
        q, k, v, output_gate, erase, write, log_decay = self._project(hidden_states)
        query, key = self._prepare_qk(q, k, erase)
        if commit:
            read, next_state = token_memory_packed(
                query,
                key,
                v,
                log_decay,
                write,
                read_states,
                write_states,
                lengths=lengths,
                scale=self.read_scale,
                chunk_tokens=self.chunk_tokens,
            )
        else:
            read = packed_snapshot_read(query, read_states, lengths, self.read_scale)
            next_state = None
        return self.o_norm(read, output_gate).reshape(
            len(hidden_states), self.value_dim
        ), next_state

    def _prepare_qk(self, q, k, erase):
        """Normalize Q/K and apply the stable erase gate."""

        return resolve_stable_qk()(q, k, erase)

    def forward(self, hidden_states, initial_state, *, commit):
        """Single-request entry; serving calls packed() for batched samples."""
        return self.packed(
            hidden_states,
            initial_state,
            lengths=[len(hidden_states)],
            commit=commit,
            write_states=initial_state,
        )
