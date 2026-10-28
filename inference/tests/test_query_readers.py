"""CPU projections/readers with explicit substitutes for CUDA-only kernels.

These tests execute the source modules and real Torch tensors, not a source-text
model. They do not claim Triton numerical or asynchronous CUDA verification.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from test_query_plan_cache import cuda_metadata as cuda_metadata
from torch.nn import functional as F

from memt_sglang.models.common.constants import WY_TILE_TOKENS
from memt_sglang.models.common.packed import boundaries
from memt_sglang.models.deepseek.ops.read import packed_snapshot_read, snapshot_read
from memt_sglang.models.qwen.ops.layout import build_chunk_plan

ROOT = Path(__file__).resolve().parents[1] / "memt_sglang"
plans = importlib.import_module("memt_sglang.models.qwen.ops.query_plan")


def load_source(relative):
    spec = importlib.util.spec_from_file_location(
        "_cpu_test_" + relative.replace("/", "_").replace(".", "_"), ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_normalize(value):
    work = value.float()
    return (work * torch.rsqrt(work.square().sum(-1, keepdim=True) + 1e-6)).to(
        value.dtype
    )


def cpu_tiled_read(q, states, plan, scale, out_dtype, *, state_slots=None):
    output = torch.empty((1, q.shape[1], q.shape[2], states.shape[-1]), dtype=out_dtype)
    for start, length, sample in zip(
        plan.chunk_tok0.tolist(),
        plan.chunk_len.tolist(),
        plan.chunk_block_ids.tolist(),
        strict=True,
    ):
        slot = sample if state_slots is None else int(state_slots[sample])
        for head in range(q.shape[2]):
            value = torch.zeros(length, states.shape[-1], dtype=torch.float32)
            for offset in range(0, q.shape[-1], 64):
                value += (
                    q[0, start : start + length, head, offset : offset + 64].float()
                    @ states[slot, head, offset : offset + 64].to(q.dtype).float()
                )
            output[0, start : start + length, head] = (value * scale).to(out_dtype)
    return output


def cpu_validate(block, hidden, reads, writes, lengths):
    # Only the CUDA-device requirement is replaced at the CPU test boundary.
    assert hidden.ndim == 2 and hidden.shape[1] == block.hidden_size
    assert hidden.dtype == torch.bfloat16 and sum(lengths) == len(hidden)
    for state in (reads, writes):
        assert state.shape == (
            len(lengths),
            block.num_heads,
            block.head_dim,
            block.head_dim,
        )
        assert state.dtype == torch.float32 and state.device == hidden.device


@pytest.fixture
def cpu_modules(monkeypatch):
    def substitute(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    writer = Mock(side_effect=AssertionError("query entered writer"))
    substitute("memt_sglang.models.qwen.ops.normalization", l2norm=cpu_normalize)
    substitute("memt_sglang.models.qwen.ops.packed", token_memory_packed=writer)
    substitute("memt_sglang.models.qwen.ops.read", memt_gdn2_read_fwd=cpu_tiled_read)
    substitute(
        "memt_sglang.models.qwen.ops.slot_read",
        slot_query_supported=lambda *args: True,
        slot_snapshot_read=cpu_tiled_read,
    )
    substitute("memt_sglang.models.deepseek.ops.packed", token_memory_packed=writer)
    substitute("transformers.activations", ACT2FN={"silu": F.silu})
    tensor_ops = load_source("models/deepseek/ops/compiled_tensor_ops.py")
    substitute(
        "memt_sglang.models.deepseek.ops.compiled_tensor_ops",
        resolve_grouped_qkv_expansion=lambda: tensor_ops.expand_grouped_qkv,
        resolve_head_expansion=lambda: tensor_ops.expand_head_projections,
        resolve_stable_qk=lambda: tensor_ops.prepare_stable_qk,
        resolve_gated_rms_norm=lambda: tensor_ops._gated_rms_norm,
    )
    nano = load_source("models/qwen/layer.py")
    flash = load_source("models/deepseek/layer.py")
    monkeypatch.setattr(nano, "validate_packed_inputs", cpu_validate)
    monkeypatch.setattr(flash, "validate_packed_inputs", cpu_validate)
    return NS(nano=nano, flash=flash, writer=writer)


@pytest.fixture
def blocks(cpu_modules):
    torch.manual_seed(20261027)
    nano = cpu_modules.nano.QwenMemoryBlock(
        NS(
            hidden_size=12,
            num_heads=2,
            head_dim=4,
            chunk_tokens=2048,
            rms_norm_eps=1e-6,
            output_gate_softcap=2.0,
        ),
        device="cpu",
    )
    flash = cpu_modules.flash.DeepSeekMemoryBlock(
        NS(
            hidden_size=12,
            num_heads=2,
            head_dim=4,
            chunk_tokens=2048,
            kv_groups=2,
            q_rank=4,
            kv_rank=4,
            gate_rank=4,
            norm_eps=1e-6,
            output_softcap=2.0,
        ),
        device="cpu",
    )
    with torch.no_grad():
        flash.k_group_up.normal_(std=0.1)
        flash.v_group_up.normal_(std=0.1)
    return NS(nano=nano, flash=flash)


def legacy_nano(block, hidden, states, lengths):
    projected = F.linear(hidden.unsqueeze(0), block.in_proj.weight[: block.qkv_width])
    q, _, _ = projected.split(block.qkv_rows, dim=-1)
    q = cpu_normalize(
        q.view(1, len(hidden), block.num_heads, block.head_dim).contiguous()
    )
    gate = block.g_down(
        F.linear(hidden.unsqueeze(0), block.in_proj.weight[-block.head_dim :])
    )
    bounds = boundaries(lengths, "cpu")
    plan = build_chunk_plan(
        bounds,
        bounds,
        WY_TILE_TOKENS,
        event_slice=bounds,
        commit_mask=torch.zeros(len(lengths), dtype=torch.bool),
    )
    read = cpu_tiled_read(q, states, plan, block.read_scale, hidden.dtype)
    return block.o_norm(
        read.reshape(-1, block.head_dim), gate.reshape(-1, block.head_dim)
    ).view(len(hidden), block.value_dim)


@pytest.mark.parametrize("lengths", [(1,), (1,) * 4, (63, 64, 65), (1, 3), (2, 2)])
@pytest.mark.parametrize("route", ["plain", "slots", "fallback"])
def test_nano_all_query_entries_use_plan_and_match_legacy(
    blocks, cpu_modules, monkeypatch, lengths, route
):
    block = blocks.nano
    hidden = torch.randn(sum(lengths), 24, dtype=torch.bfloat16)[:, ::2]
    pool = torch.randn(5, 2, 4, 4)
    before = pool.clone()
    slots = torch.tensor([4, 1, 4, 0][: len(lengths)])
    expected = legacy_nano(block, hidden, pool[slots], lengths)
    provider = Mock(side_effect=plans.query_plan)
    monkeypatch.setattr(cpu_modules.nano, "query_plan", provider)
    projection = Mock(wraps=F.linear)
    monkeypatch.setattr(F, "linear", projection)
    monkeypatch.setattr(
        cpu_modules.nano, "slot_query_supported", lambda *a: route != "fallback"
    )
    if route == "plain":
        actual, final = block.packed(
            hidden,
            pool[slots],
            lengths=lengths,
            commit=False,
            write_states=pool[slots],
        )
    else:
        actual, final = block.packed_query_slots(hidden, pool, slots, lengths=lengths)
    provider.assert_called_once_with(lengths, hidden.device)
    assert projection.call_args_list[0].args[1].shape == (24, 12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, before, rtol=0, atol=0)
    assert final is None
    cpu_modules.writer.assert_not_called()


@pytest.mark.parametrize("lengths", [(1,), (1,) * 4, (2, 1, 5)])
@pytest.mark.parametrize("route", ["plain", "slots", "fallback"])
def test_flash_query_dispatch_and_math(
    blocks, cpu_modules, monkeypatch, lengths, route
):
    block = blocks.flash
    hidden = torch.randn(sum(lengths), 24, dtype=torch.bfloat16)[:, ::2]
    pool = torch.randn(5, 2, 4, 4)
    slots = torch.tensor([4, 1, 4, 0][: len(lengths)])
    before = pool.clone()
    q, k, _, gate, erase, _, _ = block._project(hidden)
    q, _ = block._prepare_qk(q, k, erase)
    expected = block.o_norm(
        snapshot_read(
            q, pool[slots], lengths, block.read_scale, bf16_accumulator_output=True
        ),
        gate,
    ).reshape(-1, 8)
    fast = Mock(wraps=packed_snapshot_read)
    monkeypatch.setattr(cpu_modules.flash, "packed_snapshot_read", fast)
    gather = Mock(
        side_effect=lambda state, ids: state.index_select(0, ids).to(torch.bfloat16)
    )
    module = importlib.import_module("memt_sglang.models.deepseek.ops.slot_read")
    monkeypatch.setattr(module, "gather_bf16", gather)
    if route == "slots":
        # Exercise the serving dispatch predicate with CPU storage; no CUDA runs.
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    if route == "plain":
        actual, final = block.packed(
            hidden, pool[slots], lengths=lengths, commit=False, write_states=pool[slots]
        )
    else:
        actual, final = block.packed_query_slots(hidden, pool, slots, lengths=lengths)
    assert fast.call_count == 1
    assert gather.call_count == int(route == "slots")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, before, rtol=0, atol=0)
    assert final is None
    cpu_modules.writer.assert_not_called()


@pytest.mark.parametrize("lengths", [(1,), (1,) * 32, (2, 1, 5)])
def test_flash_reader_keeps_bf16_accumulation_and_noncontiguous_query(
    lengths, monkeypatch
):
    query = torch.randn(1, sum(lengths), 2, 8, dtype=torch.bfloat16)[..., ::2]
    states = torch.randn(len(lengths), 2, 4, 4)
    expected = snapshot_read(query, states, lengths, 0.5, bf16_accumulator_output=True)
    product = Mock(wraps=torch.baddbmm)
    monkeypatch.setattr(torch, "baddbmm", product)
    actual = packed_snapshot_read(query, states, lengths, 0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert product.call_count == 1
    assert product.call_args.kwargs == {"beta": 0, "alpha": 0.5}


def test_nano_launch_accepts_reader_only_plan(monkeypatch):
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def run(**kw):
                launches.append((grid, kw))
                plan = NS(
                    chunk_tok0=kw["chunk_tok0"],
                    chunk_len=kw["chunk_len"],
                    chunk_block_ids=kw["chunk_block_ids"],
                )
                kw["o"].copy_(
                    cpu_tiled_read(
                        kw["q"], kw["snaps"], plan, kw["scale"], kw["o"].dtype
                    )
                )

            return run

    monkeypatch.setitem(sys.modules, "triton", NS(cdiv=lambda a, b: (a + b - 1) // b))
    monkeypatch.setitem(
        sys.modules,
        "memt_sglang.models.qwen.ops._triton.read",
        NS(BK_READ=64, BV_READ=64, _read_fwd_kernel=Kernel()),
    )
    module = load_source("models/qwen/ops/read.py")
    plan = plans.query_plan((1, 3), "cpu")
    q, state = torch.randn(1, 4, 2, 4, dtype=torch.bfloat16), torch.randn(2, 2, 4, 4)
    actual = module.memt_gdn2_read_fwd(q, state, plan, 0.5, torch.bfloat16)
    torch.testing.assert_close(
        actual, cpu_tiled_read(q, state, plan, 0.5, q.dtype), rtol=0, atol=0
    )
    assert launches[0][0] == (2, 2, 1)
    assert launches[0][1]["chunk_tok0"] is plan.chunk_tok0


@pytest.mark.parametrize("model", ["nano", "flash"])
def test_commit_preserves_writer_contract(blocks, cpu_modules, monkeypatch, model):
    block = getattr(blocks, model)
    hidden, states = torch.randn(2, 12, dtype=torch.bfloat16), torch.randn(1, 2, 4, 4)
    before = states.clone()
    next_state = states + 1
    cpu_modules.writer.side_effect = lambda *args, **kw: (args[0].clone(), next_state)
    monkeypatch.setattr(
        cpu_modules.nano, "query_plan", Mock(side_effect=AssertionError)
    )
    _, final = block.packed(
        hidden, states, lengths=(2,), commit=True, write_states=states
    )
    assert final is next_state
    call = cpu_modules.writer.call_args
    assert call.args[5] is states and call.args[6] is states
    assert call.kwargs["lengths"] == (2,)
    assert call.kwargs["chunk_tokens"] == 2048
    torch.testing.assert_close(states, before, rtol=0, atol=0)


@pytest.mark.parametrize("model", ["nano", "flash"])
def test_standalone_forward_remains_available(blocks, model):
    block = getattr(blocks, model)
    result, final = block(
        torch.randn(1, 12, dtype=torch.bfloat16), torch.randn(1, 2, 4, 4), commit=False
    )
    assert result.shape == (1, 8) and final is None


def test_nano_entries_share_geometry_but_read_current_slots(
    blocks, cpu_modules, cuda_metadata, monkeypatch
):
    block = blocks.nano
    monkeypatch.setattr(
        cpu_modules.nano,
        "query_plan",
        lambda lengths, device: plans.query_plan(lengths, "cuda:0"),
    )
    hidden = torch.randn(4, 12, dtype=torch.bfloat16)
    pool = torch.randn(5, 2, 4, 4)
    before = pool.clone()
    for route, ids in (("plain", [4, 1]), ("slots", [1, 4]), ("fallback", [2, 2])):
        slots = torch.tensor(ids)
        monkeypatch.setattr(
            cpu_modules.nano, "slot_query_supported", lambda *args: route != "fallback"
        )
        if route == "plain":
            actual, final = block.packed(
                hidden,
                pool[slots],
                lengths=(1, 3),
                commit=False,
                write_states=pool[slots],
            )
        else:
            actual, final = block.packed_query_slots(
                hidden, pool, slots, lengths=(1, 3)
            )
        torch.testing.assert_close(
            actual, legacy_nano(block, hidden, pool[slots], (1, 3)), rtol=0, atol=0
        )
        assert final is None
    assert len(cuda_metadata.creations) == 3
    assert plans._device_plan.cache_info().hits == 2
    torch.testing.assert_close(pool, before, rtol=0, atol=0)
