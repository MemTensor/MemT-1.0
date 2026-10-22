"""Full-head replicated memory; numerical methods come from the release."""

import torch
from torch import nn
from torch.nn import functional as F

from memt_sglang.models.common.constants import WY_TILE_TOKENS
from memt_sglang.models.common.packed import boundaries, validate_packed_inputs
from memt_sglang.models.qwen.ops.layout import build_chunk_plan
from memt_sglang.models.qwen.ops.normalization import l2norm
from memt_sglang.models.qwen.ops.packed import token_memory_packed
from memt_sglang.models.qwen.ops.query_plan import query_plan
from memt_sglang.models.qwen.ops.read import memt_gdn2_read_fwd
from memt_sglang.models.qwen.ops.slot_read import (
    slot_snapshot_read,
    slot_query_supported,
)


class MemTRMSNormGated(nn.Module):
    def __init__(self, hidden_size, *, eps, output_gate_softcap_limit):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.output_gate_softcap_limit = float(output_gate_softcap_limit)

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        gate_value = F.silu(gate.to(torch.float32))
        limit = self.output_gate_softcap_limit
        gate_value = limit * torch.tanh(gate_value / limit)
        return (hidden_states * gate_value).to(input_dtype)


class QwenMemoryBlock(nn.Module):
    def __init__(self, memory_config, *, device=None):
        super().__init__()
        hidden_size, num_heads, head_dim = (
            memory_config.hidden_size,
            memory_config.num_heads,
            memory_config.head_dim,
        )
        self.chunk_tokens = memory_config.chunk_tokens
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = num_heads * head_dim
        self.read_scale = head_dim**-0.5
        self.qkv_rows = (num_heads * head_dim,) * 3
        erase_write_rows = (num_heads * head_dim,) * 2
        self.gate_rows = (*erase_write_rows, head_dim, head_dim)
        self.in_proj_rows = self.qkv_rows + self.gate_rows
        self.qkv_width = sum(self.qkv_rows)
        self.write_gate_rows = self.gate_rows[:-1]
        self._write_gate_slice = slice(
            self.qkv_width, self.qkv_width + sum(self.write_gate_rows)
        )
        self.in_proj = nn.Linear(hidden_size, sum(self.in_proj_rows), bias=False)
        self.f_down = nn.Linear(head_dim, num_heads * head_dim, bias=False)
        self.g_down = nn.Linear(head_dim, num_heads * head_dim, bias=False)
        self.decay_scale_log = nn.Parameter(torch.zeros(num_heads))
        self.decay_bias = nn.Parameter(torch.zeros(num_heads * head_dim))
        self.o_norm = MemTRMSNormGated(
            head_dim,
            eps=memory_config.rms_norm_eps,
            output_gate_softcap_limit=memory_config.output_gate_softcap,
        )
        # All parameters are replaced by strict HF loading before model attachment.
        self.to(device=device, dtype=torch.bfloat16)
        self.eval()

    def _log_decay(self, decay_latent):
        gate = F.softplus(self.f_down(decay_latent).float() + self.decay_bias)
        return -self.decay_scale_log.float().exp().view(self.num_heads, 1) * gate.view(
            *gate.shape[:-1], self.num_heads, -1
        )

    def _write_gates(self, gates, batch, seq_len):
        num_heads, head_dim = self.num_heads, self.head_dim
        *write_gates, decay_latent = torch.split(gates, self.write_gate_rows, dim=-1)
        log_decay = self._log_decay(decay_latent)
        erase = (
            torch.exp(0.5 * F.logsigmoid(write_gates[0].float()))
            .to(write_gates[0].dtype)
            .view(batch, seq_len, num_heads, head_dim)
        )
        write = write_gates[1].sigmoid().view(batch, seq_len, num_heads, head_dim)
        log_decay = log_decay.view(batch, seq_len, num_heads, -1)
        return erase, write, log_decay

    @staticmethod
    def _symmetric_inputs(q, k, erase_sqrt):
        query_normalized, key_normalized = (
            l2norm(q.contiguous()),
            l2norm(k.contiguous()),
        )
        key_scaled = (erase_sqrt.float() * key_normalized.float()).to(k.dtype)
        return query_normalized, key_scaled, torch.ones_like(erase_sqrt)

    @torch.no_grad()
    def packed(self, hidden_states, read_states, *, lengths, commit, write_states):
        validate_packed_inputs(self, hidden_states, read_states, write_states, lengths)
        hidden_batch = hidden_states.unsqueeze(0)
        weight, dim = self.in_proj.weight, self.head_dim
        q, k, v = F.linear(hidden_batch, weight[: self.qkv_width]).split(
            self.qkv_rows, dim=-1
        )
        q, k, v = [
            projection.view(1, len(hidden_states), self.num_heads, dim)
            for projection in (q, k, v)
        ]
        output_gate = self.g_down(F.linear(hidden_batch, weight[-dim:]))
        if not commit:
            samples = boundaries(lengths, hidden_states.device)
            # Read fixed snapshots with the released tiled reduction order.
            # Preserve QKV GEMM layout; a q-only GEMM can change rounding.
            q = l2norm(q.contiguous())
            mask = torch.zeros(
                len(lengths), device=hidden_states.device, dtype=torch.bool
            )
            plan = build_chunk_plan(
                samples,
                samples,
                WY_TILE_TOKENS,
                event_slice=samples,
                commit_mask=mask,
            )
            read = memt_gdn2_read_fwd(
                q, read_states, plan, self.read_scale, out_dtype=v.dtype
            )
            output = self.o_norm(read.reshape(-1, dim), output_gate.reshape(-1, dim))
            return output.view(len(hidden_states), self.value_dim), None
        erase, write, log_decay = self._write_gates(
            F.linear(hidden_batch, weight[self._write_gate_slice]),
            1,
            len(hidden_states),
        )
        q, k, erase = self._symmetric_inputs(q, k, erase)
        read, next_state = token_memory_packed(
            q,
            k,
            v,
            log_decay,
            write,
            read_states,
            write_states,
            erase=erase,
            lengths=lengths,
            scale=self.read_scale,
            chunk_tokens=self.chunk_tokens,
        )
        output = self.o_norm(read.reshape(-1, dim), output_gate.reshape(-1, dim))
        return output.view(len(hidden_states), self.value_dim), next_state

    @torch.no_grad()
    def packed_query_slots(self, hidden_states, state_pool, slots, *, lengths):
        if not slot_query_supported(self, hidden_states, state_pool, slots, lengths):
            states = state_pool.index_select(0, slots)
            return self.packed(hidden_states, states, lengths=lengths,
                               commit=False, write_states=states)
        # Intentionally mirror public packed()'s full QKV GEMM and view layout.
        # A q-only projection changes BF16 rounding despite discarding K and V.
        hidden_batch = hidden_states.unsqueeze(0)
        weight, dim = self.in_proj.weight, self.head_dim
        q, k, v = F.linear(hidden_batch, weight[: self.qkv_width]).split(
            self.qkv_rows, dim=-1
        )
        q, k, v = [
            projection.view(1, len(hidden_states), self.num_heads, dim)
            for projection in (q, k, v)
        ]
        output_gate = self.g_down(F.linear(hidden_batch, weight[-dim:]))
        q = l2norm(q.contiguous())
        plan = query_plan(lengths, hidden_states.device)
        read = slot_snapshot_read(
            q, state_pool, plan, self.read_scale, out_dtype=v.dtype, state_slots=slots
        )
        output = self.o_norm(read.reshape(-1, dim), output_gate.reshape(-1, dim))
        return output.view(len(hidden_states), self.value_dim), None

    def forward(self, hidden_states, initial_state, *, commit):
        """Single-request entry; serving calls packed() for batched samples."""
        return self.packed(
            hidden_states,
            initial_state,
            lengths=[len(hidden_states)],
            commit=commit,
            write_states=initial_state,
        )
