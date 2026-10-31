"""Explicit 4096-token candidate admission, never CUDA qualification."""

from types import SimpleNamespace as NS

import pytest
from test_flash_prefill import config_args, native_eligibility, runner_and_batch

from memt_sglang.runtime import flash_prefill as fp


@pytest.mark.parametrize("tokens", [(4096,), (512, 4096), (1024, 4096), (512, 1024, 4096)])
def test_explicit_4096_config_and_resolution_remain_unqualified(tokens):
    intent = fp.FlashPrefillIntent("breakable", tokens, 4)
    phase = fp.explicit_capture_config(intent)
    assert phase == {"backend": "breakable", "bs": list(tokens), "max_bs": 4096}
    args, _ = config_args()
    args.cuda_graph_config.prefill = NS(**phase)
    receipt = fp.validate_capture_request(args, intent, native_validator=lambda a: None)
    assert receipt["requested_capture_tokens"] == list(tokens)
    assert receipt["resolved_capture_tokens"] == list(tokens)
    assert receipt["status"] == "capture_attempt_permitted"
    assert receipt["gpu_qualified"] is False
    assert "4096" in receipt["replay_scope"]


@pytest.mark.parametrize("tokens", [
    (0,), (256,), (2048,), (4095,), (4097,), (8192,), (True,), (4096.0,),
    (4096, 512), (512, 512), (),
])
def test_unreviewed_or_invalid_gears_remain_rejected(tokens):
    with pytest.raises(ValueError):
        fp.explicit_capture_config(fp.FlashPrefillIntent("breakable", tokens, 4))


def bs32_vote(rank, counts=(4096,) * 4):
    runner, batch = runner_and_batch(rank=rank, counts=counts)
    runner.capture_num_tokens = [512, 1024, 4096]
    runner.max_num_tokens = 4096
    record = runner.model_runner.model.memt_runtime.worker_sessions.requests["q"]
    requests = {f"q{i}": record for i in range(8)}
    runner.model_runner.model.memt_runtime.worker_sessions.requests = requests
    batch.rids = list(requests)
    batch.batch_size = 8
    batch.extend_seq_lens_cpu = [512] * 8
    batch.extend_prefix_lens_cpu = [0] * 8
    return runner, batch


def test_bs32_eight_requests_per_rank_admits_exact_4096_with_native_delegate():
    votes = [fp.replay_vote(*bs32_vote(rank), rank=rank, native_can_run=native_eligibility())
             for rank in range(4)]
    agreement = fp.agree_replay(votes, dp_size=4)
    assert agreement.allowed and agreement.bucket == 4096
    assert all(v.original_counts == (4096,) * 4 for v in votes)


@pytest.mark.parametrize("counts,reason", [
    ((4096, 4096, 4096, 0), "original_dp_idle"),
    ((4096, 4096, 4096, 3584), "original_dp_uneven"),
])
def test_4096_never_relaxes_original_idle_or_uneven_veto(counts, reason):
    runner, batch = bs32_vote(0, counts)
    vote = fp.replay_vote(runner, batch, rank=0, native_can_run=native_eligibility())
    assert reason in vote.reasons


def test_4096_native_veto_and_uncaptured_2048_still_fall_back():
    runner, batch = bs32_vote(0)
    batch.input_embeds = object()
    assert fp.replay_vote(runner, batch, rank=0, native_can_run=native_eligibility()).reasons == ("native_veto",)
    runner, batch = runner_and_batch(counts=(2048,) * 4)
    runner.capture_num_tokens = [512, 1024, 4096]
    runner.max_num_tokens = 4096
    assert "dp_bucket_padding_unverified" in fp.replay_vote(
        runner, batch, rank=0, native_can_run=native_eligibility()).reasons


def test_capture_dummy_must_fit_context_capacity_before_scratch_allocation(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runner, _ = runner_and_batch(counts=(4096,) * 4)
    runner.capture_num_tokens = [512, 1024, 4096]
    runner.require_mlp_tp_gather = True
    runner.backend = NS(_debug_eager=False)
    attn = runner.model_runner.attn_backend
    attn.page_size, attn.c4_topk = 256, 512
    attn.req_to_token = NS(shape=(32, 2048))
    attn.MAX_SEQ_LEN_FOR_CAPTURE = 2048
    with pytest.raises(ValueError, match="gear exceeds"):
        fp._worker_preflight(runner, fp.FlashPrefillIntent("breakable", (512, 1024, 4096), 4),
                             lambda: pytest.fail("context veto must precede memory observation"))
