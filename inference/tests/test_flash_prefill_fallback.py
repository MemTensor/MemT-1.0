"""A rejected Flash gear must retain native eager DP padding before normalization."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from test_flash_prefill import runner_and_batch

from memt_sglang.runtime import flash_prefill as fp
from memt_sglang.runtime.flash_prefill import FlashPrefillIntent


def install(counts, *, rank=0, flash=True, extend=True, enabled=True, captured=True):
    calls = []

    class Runner:
        def can_run_graph(self, batch):
            return True

    class ModelRunner:
        def _prepare_eager_forward_batch(self, batch):
            # Native prepare_mlp_sync_batch forces MAX_LEN for a possible
            # breakable prefill before the post-normalization replay decision.
            force = batch.can_run_dp_breakable_cuda_graph and batch.is_extend_in_batch
            original = batch.global_num_tokens_cpu
            effective = [max(original)] * len(original) if force else original
            calls.append((force, tuple(effective)))
            batch.global_num_tokens_cpu = effective
            batch.global_dp_buffer_len = sum(effective)
            batch.dp_padding_mode = NS(is_max_len=lambda: force)

    gathered = []

    def gather(vote):
        gathered.append(vote)
        return [replace(vote, rank=i) for i in range(4)]

    fp._install_replay_consensus(
        fp.FlashPrefillIntent("breakable", (512, 1024, 4096), 4),
        Runner, ModelRunner, gather, lambda: rank, lambda event: None,
    )
    runner, batch = runner_and_batch(rank=rank, counts=counts)
    runner.capture_num_tokens = [512, 1024, 4096]
    runner._memt_flash_capture_complete = captured
    batch.global_num_tokens_cpu = list(counts)
    batch.can_run_dp_breakable_cuda_graph = enabled
    batch.is_extend_in_batch = extend
    batch.input_ids = [7] * counts[rank]
    if not counts[rank]:
        batch.rids = []
        batch.batch_size = 0
        batch.extend_seq_lens_cpu = []
        batch.extend_prefix_lens_cpu = []
        batch.forward_mode = NS(is_idle=lambda: True, is_extend=lambda: False,
                                is_decode=lambda: False)
        batch._memt_graph_allowed = True
    mr = ModelRunner()
    mr.model_config = NS(hf_config=NS(model_type="memt_deepseek_v4" if flash else "other"))
    mr.prefill_cuda_graph_runner = runner
    return mr, batch, calls, gathered


@pytest.mark.parametrize("counts", [
    (512, 0, 0, 0), (0, 512, 512, 512), (512, 512, 512, 0),
    (1024, 512, 512, 512), (511, 511, 511, 511),
    (513, 513, 513, 513), (2048, 2048, 2048, 2048),
])
@pytest.mark.parametrize("rank", range(4))
def test_fallback_veto_precedes_native_padding_on_every_rank(counts, rank):
    mr, batch, calls, gathered = install(counts, rank=rank)
    mr._prepare_eager_forward_batch(batch)
    assert calls == [(False, counts)]
    assert batch.global_num_tokens_cpu == list(counts)
    assert batch.original_global_num_tokens_cpu == list(counts)
    assert batch._memt_flash_prefill_allowed is False
    assert len(gathered) == 1


def test_effective_counts_mismatch_cannot_force_graph_padding():
    mr, batch, calls, gathered = install((512,) * 4)
    batch.global_num_tokens_cpu = [512, 511, 512, 512]
    mr._prepare_eager_forward_batch(batch)
    assert calls == [(False, (512, 511, 512, 512))]
    assert batch._memt_flash_prefill_allowed is False
    assert len(gathered) == 1


@pytest.mark.parametrize("tokens", [512, 1024, 4096])
def test_exact_balanced_gears_keep_native_graph_normalization(tokens):
    counts = (tokens,) * 4
    mr, batch, calls, gathered = install(counts)
    mr._prepare_eager_forward_batch(batch)
    assert calls == [(True, counts)]
    assert batch._memt_flash_prefill_allowed is True
    assert len(gathered) == 1


def test_existing_native_veto_is_never_enabled():
    mr, batch, calls, _ = install((512,) * 4, enabled=False)
    mr._prepare_eager_forward_batch(batch)
    assert calls == [(False, (512,) * 4)]
    assert batch.can_run_dp_breakable_cuda_graph is False


@pytest.mark.parametrize("options", [dict(flash=False), dict(extend=False), dict(captured=False)])
def test_unrelated_paths_keep_the_original_preparation(options):
    mr, batch, calls, gathered = install((512, 0, 0, 0), **options)
    mr._prepare_eager_forward_batch(batch)
    assert batch.can_run_dp_breakable_cuda_graph is True
    assert len(calls) == 1
    assert gathered == []


@pytest.mark.parametrize("original", [None, [], [512, 512], [512, 512, True, 512]])
def test_unknown_original_counts_cannot_force_graph_padding(original):
    mr, batch, calls, gathered = install((512,) * 4)
    batch.original_global_num_tokens_cpu = original
    mr._prepare_eager_forward_batch(batch)
    assert calls[0][0] is False
    assert batch._memt_flash_prefill_allowed is False
    assert len(gathered) == 1


def test_decode_round_never_enters_prefill_collective_or_overrides_query_vote():
    class Runner:
        def can_run_graph(self, batch): return True
    class ModelRunner:
        def _prepare_eager_forward_batch(self, batch): pass
    fp._install_replay_consensus(FlashPrefillIntent("breakable", (512,), 4),
                                 Runner, ModelRunner,
                                 lambda v: pytest.fail("decode cannot gather here"),
                                 lambda: 0, lambda e: None)
    mr = ModelRunner()
    mr.model_config = NS(hf_config=NS(model_type="memt_deepseek_v4"))
    mr.prefill_cuda_graph_runner = NS(_memt_flash_capture_complete=True)
    batch = NS(is_extend_in_batch=False, _memt_graph_allowed=True,
               _memt_flash_prefill_allowed=True)
    mr._prepare_eager_forward_batch(batch)
    assert batch._memt_graph_allowed is True
    assert batch._memt_flash_prefill_allowed is False
