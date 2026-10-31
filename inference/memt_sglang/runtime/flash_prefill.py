"""Explicit DP4 Flash prefill capture gate and stable metadata repair.

Integration (launcher/worker wiring is deliberately separate):
1. Save FlashPrefillIntent BEFORE native ServerArgs resolution. Report both
   requested and resolved values AFTER resolution with prefill_identity().
2. Before DP normalization, veto graph padding for globally ineligible token
   counts so fallback keeps native eager shapes. After normalization and before
   graph/eager divergence, every DP rank
   produces replay_vote(), delegating native eligibility at most once when
   local prerequisites pass. All-gather those votes
   on the same CPU group on EVERY rank, including idle ranks. agree_replay()
   validates the complete rank set; restrict_batch() only adds a veto.
   Never add that collective inside a branch that only active ranks enter.

Use explicit_capture_config + validate_capture_request in the parent and
install_native_worker in every fresh worker. Native explicit JSON already
skips blanket auto-disable; this module rechecks all other audited native
predicates and bounds the experiment to explicit 512/1024/4096, DP4, A2A none.
The launcher defaults to 512/1024; 4096 requires an explicit gear request and
independent headroom, capture, replay and numerical checks, including any decode
graphs used alongside it. No default benefit or CUDA qualification is implied.
The native installation is not edited.
The pinned runner copies global_dp_buffer_len
from the live batch while bucket-padding input_ids (load_batch:925,1030).
Until that contract is adapted, only equal, unpadded DP token shapes qualify
for a local replay vote. Original counts are required: ForwardBatch MAX_LEN
padding can turn an idle rank into a dummy EXTEND (1277-1324), hiding zero
counts from the native runner's inactive-rank check (589-598).

Native keeps capture metadata and graph pool addresses alive. Replay repairs
the core/indexer page-table alias. Total capture-pool peak and numeric quality
must be measured by actual capture/replay. Bucket-expanded DP buffers remain
outside this initial scope. CPU tests do not qualify Torch 2.11 CUDA execution.

Production replay uses host counters only, with no added device synchronization
or per-batch logging. replay_observer(event) is an explicit external diagnostic
hook; it may synchronize on replay_return and must label that run accordingly.
Capture still synchronizes once before publishing its completion evidence.
"""

import ast
import hashlib
import inspect
import os
import textwrap
from copy import copy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from .decode_graph import query_graph_allowed


def _counts(values):
    if not isinstance(values, (list, tuple)):
        return ()
    if any(type(n) is not int or n < 0 for n in values):
        return ()
    return tuple(values)


def _buckets(values):
    result = _counts(values)
    valid = result and result[0] > 0 and tuple(sorted(set(result))) == result
    return result if valid else ()


@dataclass(frozen=True)
class FlashPrefillIntent:
    """Trusted MemT-Flash request, saved before native config can disable it."""

    requested_backend: str
    capture_tokens: tuple[int, ...]
    dp_size: int

    def __post_init__(self):
        if self.requested_backend not in ("disabled", "breakable"):
            raise ValueError("Flash prefill requires disabled or breakable")
        if type(self.dp_size) is not int or self.dp_size < 1:
            raise ValueError("dp_size must be a positive integer")
        if type(self.capture_tokens) is not tuple or (
            self.requested_backend == "breakable" and not _buckets(self.capture_tokens)
        ):
            raise ValueError("Explicit sorted positive capture tokens are required")


def prefill_identity(intent, resolved_prefill):
    """Report native resolution honestly; never re-enable or claim GPU success."""
    resolved_tokens = tuple(resolved_prefill.bs or ())
    resolved_backend = resolved_prefill.backend
    requested = intent.requested_backend == "breakable"
    if not requested:
        status = "not_requested"
    elif resolved_backend == "disabled":
        status = "disabled_by_native"
    else:
        status = "resolved_unqualified"
    pending = (
        "capture_pool_bound", "c4_scratch_lifetime",
        "dp_collective_padding", "cuda_metadata_state_validation",
    ) if requested else ()
    if requested and resolved_tokens != intent.capture_tokens:
        pending += ("resolved_shapes_changed",)
    return {
        "model": "memt-flash",
        "requested_backend": intent.requested_backend,
        "resolved_backend": resolved_backend,
        "requested_capture_tokens": list(intent.capture_tokens),
        "resolved_capture_tokens": list(resolved_tokens),
        "dp_size": intent.dp_size,
        "status": status,
        "gpu_qualified": False,
        "pending": list(pending),
    }


@dataclass(frozen=True)
class ReplayVote:
    rank: int
    dp_size: int
    capture_tokens: tuple[int, ...]
    original_counts: tuple[int, ...]
    bucket: int | None
    reasons: tuple[str, ...]

    @property
    def eligible(self):
        """Necessary local conditions only; not a capture/enablement receipt."""
        return not self.reasons


@dataclass(frozen=True)
class ReplayAgreement:
    allowed: bool
    bucket: int | None
    reasons: tuple[str, ...]


def _protocol_reasons(runner, batch):
    backend = runner.model_runner.attn_backend
    runtime = getattr(runner.model_runner.model, "memt_runtime", None)
    hooks = (
        "init_forward_metadata_for_breakable_cuda_graph_capture",
        "prepare_forward_metadata_for_breakable_cuda_graph_replay",
    )
    if runtime is None or not query_graph_allowed(runtime, batch, "prefill"):
        return ("memt_query_veto",)
    original_mode = getattr(batch, "_original_forward_mode", None)
    if original_mode is not None and not original_mode.is_extend():
        return ("original_mode_not_extend",)
    if (
        runner.prefill_backend_name != "breakable"
        or not runner.use_captured_attn_metadata
        or not getattr(
            backend, "use_captured_forward_metadata_for_breakable_cuda_graph", False
        )
        or not all(callable(getattr(backend, name, None)) for name in hooks)
    ):
        return ("captured_metadata_protocol_missing",)
    return ()


def _batch_reasons(batch, counts, rank, dp_size, bucket):
    if len(counts) != dp_size:
        return ("original_dp_counts_missing",)
    if not all(counts):
        return ("original_dp_idle",)
    if len(set(counts)) != 1:
        return ("original_dp_uneven",)
    effective = _counts(getattr(batch, "global_num_tokens_cpu", None))
    n = len(batch.input_ids)
    if effective != counts or n != counts[rank]:
        return ("dp_token_counts_inconsistent",)
    padding_mode = getattr(batch, "dp_padding_mode", None)
    if padding_mode is None or not padding_mode.is_max_len():
        return ("dp_max_len_required",)
    if getattr(batch, "global_dp_buffer_len", None) != n * dp_size:
        return ("dp_collective_buffer_mismatch",)
    if not getattr(batch, "can_run_dp_breakable_cuda_graph", False):
        return ("native_dp_veto",)
    lengths = _counts(getattr(batch, "extend_seq_lens_cpu", None))
    prefixes = _counts(getattr(batch, "extend_prefix_lens_cpu", None))
    bs = batch.batch_size
    if (
        len(lengths) != bs
        or len(prefixes) != bs
        or len(batch.rids or []) != bs
        or not all(lengths)
        or sum(lengths) != counts[rank]
        or getattr(batch, "num_token_non_padded_cpu", None) != counts[rank]
    ):
        return ("request_token_spans_inconsistent",)
    if bucket != n:
        return ("dp_bucket_padding_unverified",)
    return ()


def replay_vote(runner, batch, *, rank, native_can_run):
    """Evaluate host metadata and retain the original eligibility delegate.

    Call on every rank before execution-path divergence. No tensor reads,
    collectives, metadata rewriting, capture or file writes occur here.
    native_can_run is the unbound original (runner, batch) callable; it must
    not point back to this adapter. Native exceptions propagate unchanged.
    """
    dp_size = runner.dp_size
    if type(dp_size) is not int or dp_size < 1:
        raise ValueError("Invalid DP size")
    if type(rank) is not int or not 0 <= rank < dp_size:
        raise ValueError("Invalid DP rank")
    tokens = _buckets(runner.capture_num_tokens)
    original = _counts(getattr(batch, "original_global_num_tokens_cpu", None))
    bucket = next((s for s in tokens if s >= len(batch.input_ids)), None)
    reasons = _protocol_reasons(runner, batch)
    if not tokens or bucket is None:
        reasons += ("capture_shape_unavailable",)
    if not reasons:
        reasons = _batch_reasons(batch, original, rank, dp_size, bucket)
    if not reasons and not native_can_run(runner, batch):
        reasons = ("native_veto",)
    return ReplayVote(rank, dp_size, tokens, original, bucket, reasons)


def agree_replay(votes, *, dp_size):
    """Validate votes all-gathered by the caller; never infer absent ranks."""
    if type(dp_size) is not int or dp_size < 1:
        raise ValueError("Invalid DP size")
    votes = tuple(votes)
    if (
        len(votes) != dp_size
        or any(not isinstance(v, ReplayVote) for v in votes)
        or {v.rank for v in votes} != set(range(dp_size))
        or any(v.dp_size != dp_size for v in votes)
    ):
        return ReplayAgreement(False, None, ("rank_votes_incomplete",))
    signatures = {(v.capture_tokens, v.original_counts, v.bucket) for v in votes}
    reasons = tuple(sorted({reason for v in votes for reason in v.reasons}))
    if len(signatures) != 1:
        reasons += ("rank_shape_disagreement",)
    if any(v.bucket is None for v in votes):
        reasons += ("capture_shape_unavailable",)
    return ReplayAgreement(not reasons, votes[0].bucket if not reasons else None, reasons)


def restrict_batch(batch, agreement):
    """Return a shallow view with an additional veto; preserve caller metadata."""
    view = copy(batch)
    view._memt_graph_allowed = bool(
        getattr(batch, "_memt_graph_allowed", True) and agreement.allowed
    )
    return view


# Only these two audited *configuration* rules are relaxed for an explicit
# experimental request. Every other predicate is executed from native source.
# Never install this modified function on ServerArgs or modify its source file.
_RULE_SOURCE_SHA = "9d5197c492584639519573c9e45f0e23c1147ccb89e2d2abcd2c404e4e875046"
_EXPERIMENT_RULES = {
    "DeepSeek-V4 (heavy capture-pool memory pressure)", "DP attention",
}
_DSV4_SOURCE_SHA = "57e6a41f005e4f28a12394addb028cb4777b0f05026a8cbcefc0319527f74ea7"
CAPTURE_TOKEN_GEARS = (512, 1024, 4096)


def explicit_capture_config(intent):
    """Native JSON phase config: locks backend, bs AND max_bs before parsing.

    Explicit backend already skips native's blanket cascade (server_args:3656).
    validate_capture_request MUST run after resolution and before construction.
    """
    if (
        intent.requested_backend != "breakable" or intent.dp_size != 4
        or not intent.capture_tokens
        or not set(intent.capture_tokens) <= set(CAPTURE_TOKEN_GEARS)
    ):
        raise ValueError("Flash capture scope is explicit DP4, 512/1024/4096 only")
    return {"backend": "breakable", "bs": list(intent.capture_tokens),
            "max_bs": max(intent.capture_tokens)}


def audited_native_validator(source, namespace):
    """Extract pinned native predicates, omitting only the two named rules."""
    # Hash the reviewed method text, not Python-version-dependent AST dumps.
    # Full-file ASTs also retain class indentation INSIDE multiline strings,
    # unlike parsing dedented inspect.getsource(). Keep both paths identical.
    canonical = textwrap.dedent(source).strip() + "\n"
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if digest != _RULE_SOURCE_SHA:
        raise ValueError(f"Flash native rule source changed; review required (sha256={digest})")
    tree = ast.parse(canonical)
    node = tree.body[0]
    rule_list = next(n.value for n in node.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "rules" for t in n.targets))
    kept = [rule for rule in rule_list.elts if rule.elts[0].value not in _EXPERIMENT_RULES]
    rule_list.elts = kept
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(tree), "<audited-native-flash-rules>", "exec"), scope)
    return scope[node.name]


def _check_capture_scope(args, intent):
    explicit_capture_config(intent)
    resolved = args._resolved()
    config = args.get_model_config()
    required = {
        "device": "cuda", "tp_size": 4, "dp_size": 4, "ep_size": 4,
        "pp_size": 1, "dcp_size": 1, "disable_overlap_schedule": True,
        "enable_two_batch_overlap": False, "enable_memory_saver": False,
        "disaggregation_mode": "null", "enable_torch_compile": False,
        "enable_hierarchical_cache": False, "cpu_offload_gb": 0,
        "enable_symm_mem": False, "enable_dsa_prefill_context_parallel": False,
        "enable_prefill_cp": False, "enable_deepseek_v4_fp4_indexer": False,
        "enable_eplb": False, "debug_cuda_graph": False,
    }
    mismatches = [key for key, value in required.items() if getattr(args, key, None) != value]
    if (
        mismatches or config.hf_config.model_type != "memt_deepseek_v4"
        or config.is_multimodal or not resolved.enable_dp_attention
        or resolved.attn_cp_size != 1 or resolved.moe_a2a_backend != "none"
        or resolved.quantization is not None
        or args.speculative_algorithm is not None
        or args.expert_distribution_recorder_mode is not None
        or args.load_format not in ("auto", "safetensors")
    ):
        raise ValueError(f"Outside Flash capture scope: {mismatches}")


def validate_capture_request(args, intent, *, native_validator=None):
    """Permit a bounded capture ATTEMPT, never restore a disabled backend.

    Invoke after prepare_server_args and again in each worker before capture.
    The native validator operates on a private config copy. OOM, capture errors
    and numerical qualification remain real experiments, not static promises.
    """
    _check_capture_scope(args, intent)
    phase = args.cuda_graph_config.prefill
    locks = getattr(args, "_cuda_graph_config_locked", set())
    if (phase.backend != "breakable" or tuple(phase.bs or ()) != intent.capture_tokens
        or phase.max_bs != max(intent.capture_tokens)
        or not {( "prefill", k) for k in ("backend", "bs", "max_bs")} <= locks):
        raise ValueError("Flash requested/resolved capture config or locks differ")
    if native_validator is None:
        method = type(args)._disable_breakable_cudagraph_if_incompatible
        native_validator = audited_native_validator(inspect.getsource(method), method.__globals__)
    private = copy(args)
    private.cuda_graph_config = copy(args.cuda_graph_config)
    private.cuda_graph_config.prefill = copy(phase)
    native_validator(private)
    if private.cuda_graph_config.prefill.backend != "breakable":
        raise ValueError("Remaining native prefill compatibility veto")
    return {**prefill_identity(intent, phase), "status": "capture_attempt_permitted",
            "relaxed_native_rules": sorted(_EXPERIMENT_RULES),
            "native_rule_source_sha256": _RULE_SOURCE_SHA,
            "replay_scope": "all DP4 ranks active, equal exact "
                            + "/".join(map(str, intent.capture_tokens)) + " tokens"}


CORE_STABLE_FIELDS = (
    "raw_out_loc", "seq_lens_casual", "positions_casual", "c4_out_loc",
    "c128_out_loc", "c4_topk_lengths_raw", "c4_topk_lengths_clamp1",
    "c4_sparse_topk_lengths", "c4_sparse_page_indices", "c4_sparse_raw_indices",
)


def _tensor_addresses(value, path, seen):
    """Host-only shape/address inspection; never read GPU tensor contents."""
    if value is None or id(value) in seen:
        return ()
    seen.add(id(value))
    if callable(getattr(value, "data_ptr", None)):
        return ((path, value.data_ptr(), tuple(value.shape), str(value.dtype)),)
    if isinstance(value, (tuple, list)):
        items = enumerate(value)
    elif isinstance(value, dict):
        items = value.items()
    else:
        items = vars(value).items() if hasattr(value, "__dict__") else ()
    return tuple(entry for key, child in items
                 for entry in _tensor_addresses(child, f"{path}.{key}", seen))


def metadata_addresses(metadata):
    core = metadata.core_attn_metadata
    values = {f"core.{name}": getattr(core, name) for name in CORE_STABLE_FIELDS}
    values.update(indexer=metadata.indexer_metadata, c4_compress=metadata.c4_compress_metadata)
    # Online c128 planner is intentionally replaceable in native replay. Its
    # pointers are not falsely classified as captured static tensors here.
    return tuple(entry for name, value in values.items()
                 for entry in _tensor_addresses(value, name, set()))


def refresh_metadata(metadata, native_refresh):
    """Fix native core/indexer page-table alias after its in-place refresh.

    native DSV4AttnMetadata replaces core.page_table (237-276) while
    PagedIndexerMetadata.copy_ preserves indexer.page_table (173-188).
    Captured indexer kernels retain the latter address. Rebind the eager core
    view to it; do not reallocate the indexer's captured input.
    """
    before = metadata_addresses(metadata)
    result = native_refresh()
    if metadata_addresses(metadata) != before:
        raise RuntimeError("Flash replay changed a captured metadata address/shape")
    metadata.core_attn_metadata.page_table = metadata.indexer_metadata.page_table
    return result


def scratch_observation(tokens, *, context_capacity, free_bytes):
    """One FP32 C4 logits tensor, NOT total model/capture-pool peak usage."""
    if any(type(n) is not int or n <= 0 for n in (tokens, context_capacity, free_bytes)):
        raise ValueError("Invalid scratch dimensions or available memory")
    compressed_columns = ((context_capacity + 255) // 256) * 64
    single_logits = tokens * compressed_columns * 4
    if free_bytes <= single_logits:
        raise ValueError("Insufficient free memory even for one C4 logits scratch")
    return {"single_c4_logits_bytes": single_logits, "free_bytes": free_bytes,
            "context_capacity": context_capacity, "total_capture_pool_bytes": None}


def capture_attempt(runner, intent, native_capture, *, emit, synchronize, memory):
    """Run real native capture; errors propagate and partial graphs never pass."""
    before = memory()
    emit({"event": "capture_enter", "tokens": list(intent.capture_tokens), **before})
    try:
        result = native_capture()
        synchronize()
        graphs = runner.backend._graphs
        segments = {key.size: len(graph._segments) for key, graph in graphs.items()}
        if (set(segments) != set(intent.capture_tokens) or not all(segments.values())
            or set(runner.attn_metadata_buffers or {}) != set(intent.capture_tokens)):
            raise RuntimeError("Incomplete Flash capture registry or metadata")
        after = memory()
        free_before, free_after = before.get("free_bytes"), after.get("free_bytes")
        sampled = (free_before, free_after) if all(
            type(n) is int for n in (free_before, free_after)
        ) else ()
        emit({"event": "capture_return", "segments": segments,
              "actual": {"backend": "breakable", "status": "captured"},
              "sampled_min_free_bytes": min(sampled) if sampled else None,
              "free_delta_bytes": free_before - free_after if sampled else None,
              "peak_scope": "process_lifetime",
              "gpu_qualified": False, "replay_validated": False, **after})
        return result
    except Exception as error:
        emit({"event": "capture_error", "error_type": type(error).__name__})
        raise


def _is_flash(model_runner):
    config = getattr(getattr(model_runner, "model_config", None), "hf_config", None)
    return getattr(config, "model_type", None) == "memt_deepseek_v4"


def _worker_preflight(runner, intent, memory):
    import torch

    mr = runner.model_runner
    attn = mr.attn_backend
    if os.environ.get("SGLANG_SCHEDULER_SKIP_ALL_GATHER", "0").lower() not in ("0", "false"):
        raise ValueError("Flash requires native scheduler all-rank synchronization")
    if torch.version.hip is not None or not torch.cuda.is_available():
        raise ValueError("Flash capture requires NVIDIA CUDA")
    if (tuple(runner.capture_num_tokens) != intent.capture_tokens
        or not runner.use_captured_attn_metadata
        or not attn.use_captured_forward_metadata_for_breakable_cuda_graph
        or not callable(attn.init_forward_metadata_for_breakable_cuda_graph_capture)
        or not callable(attn.prepare_forward_metadata_for_breakable_cuda_graph_replay)
        # Native Flash top-k geometry, unrelated to removed C4 ordering flags.
        or attn.page_size != 256 or attn.c4_topk != 512
        or getattr(mr, "hisparse_coordinator", None) is not None
        or not runner.require_mlp_tp_gather
        or getattr(runner.backend, "_debug_eager", False)):
        raise ValueError("Flash worker capture protocol/layout unsupported")
    capacity = attn.req_to_token.shape[1]
    if attn.MAX_SEQ_LEN_FOR_CAPTURE != capacity:
        raise ValueError("Capture context capacity differs from KV table")
    # Native capture_prepare places the whole gear in one dummy request, even
    # when real replay will use eight 512-token requests per rank for BS32.
    if max(intent.capture_tokens) > capacity:
        raise ValueError("Capture gear exceeds the single-request context capacity")
    source = Path(inspect.getfile(type(attn))).read_bytes()
    if hashlib.sha256(source).hexdigest() != _DSV4_SOURCE_SHA:
        raise ValueError("DSV4 source changed; metadata adapter requires review")
    return {**scratch_observation(max(intent.capture_tokens), context_capacity=capacity,
                                  free_bytes=memory()["free_bytes"]),
            "metadata_source_sha256": _DSV4_SOURCE_SHA}


def _install_capture_hook(intent, runner_cls, gather, emit, synchronize, memory,
                          replay_observer=None):
    native = runner_cls.capture

    @wraps(native)
    def capture(self):
        if not _is_flash(self.model_runner):
            return native(self)
        error = None
        try:
            receipt = validate_capture_request(self.model_runner.server_args, intent)
            scratch = _worker_preflight(self, intent, memory)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            receipt, scratch = {}, {}
        vote = {"error": error, "tokens": intent.capture_tokens, "dp_size": intent.dp_size,
                "context_capacity": scratch.get("context_capacity"),
                "metadata_source_sha256": scratch.get("metadata_source_sha256")}
        votes = gather(vote)
        if (len(votes) != intent.dp_size or any(v["error"] for v in votes)
            or any(v["tokens"] != intent.capture_tokens or v["dp_size"] != intent.dp_size for v in votes)
            or any(v.get("context_capacity") != vote["context_capacity"]
                   or v.get("metadata_source_sha256") != vote["metadata_source_sha256"] for v in votes)):
            emit({"event": "capture_preflight_rejected", "votes": votes})
            raise RuntimeError("Flash capture preflight rejected on one or more ranks")
        emit({"event": "capture_gate", **receipt, **scratch,
              "replay_observer_enabled": replay_observer is not None,
              "production_replay_synchronizes": False})
        attn = self.model_runner.attn_backend
        attn._memt_flash_metadata_adapter = True
        self._memt_flash_capture_complete = False
        result = capture_attempt(self, intent, lambda: native(self), emit=emit,
                                 synchronize=synchronize, memory=memory)
        _attach_replay_counter(self)
        self._memt_flash_capture_complete = True
        return result

    runner_cls.capture = capture


def _install_metadata_hook(attention_cls):
    native = attention_cls.prepare_forward_metadata_for_breakable_cuda_graph_replay

    @wraps(native)
    def refresh(self, metadata, batch, *args, **kwargs):
        def call():
            return native(self, metadata, batch, *args, **kwargs)

        if not getattr(self, "_memt_flash_metadata_adapter", False):
            return call()
        return refresh_metadata(metadata, call)

    attention_cls.prepare_forward_metadata_for_breakable_cuda_graph_replay = refresh


def _preserve_eager_padding(model_runner, batch, intent):
    if not _is_flash(model_runner) or getattr(batch, "is_extend_in_batch", False) is not True:
        return
    runner = getattr(model_runner, "prefill_cuda_graph_runner", None)
    if not getattr(runner, "_memt_flash_capture_complete", False):
        return
    counts = _counts(getattr(batch, "original_global_num_tokens_cpu", None))
    effective = _counts(getattr(batch, "global_num_tokens_cpu", None))
    # Native prepare_mlp_sync_batch forces MAX_LEN before the replay vote.
    # Rejecting later would retain padded dummy rows and changed GEMM/collective
    # shapes even though the batch executes eagerly. These counts are already
    # shared by every DP rank; no local eligibility or new collective belongs here.
    if (len(counts) != intent.dp_size or len(set(counts)) != 1
            or counts[0] not in intent.capture_tokens or effective != counts):
        batch.can_run_dp_breakable_cuda_graph = False


def _install_replay_consensus(intent, runner_cls, model_runner_cls, gather, rank, emit,
                              replay_observer=None):
    prepare = model_runner_cls._prepare_eager_forward_batch
    native_can_run = runner_cls.can_run_graph

    @wraps(prepare)
    def normalized(self, batch):
        _preserve_eager_padding(self, batch, intent)
        result = prepare(self, batch)
        if not _is_flash(self):
            return result
        runner = self.prefill_cuda_graph_runner
        if not getattr(runner, "_memt_flash_capture_complete", False):
            return result
        # Native dp_attn.py:186-215 broadcasts max(is_extend_in_batch) and
        # min(can_cuda_graph): any prefill disables decode graph on ALL ranks.
        # Pure decode may return early on some ranks; never gather here then.
        if getattr(batch, "is_extend_in_batch", False) is not True:
            batch._memt_flash_prefill_allowed = False
            return result
        # Runs before the native is_extend/eligibility branch on EVERY rank,
        # including ranks converted from IDLE. Existing query consensus stays.
        try:
            local = replay_vote(runner, batch, rank=rank(), native_can_run=native_can_run)
            error = None
        except Exception as exc:
            local = ReplayVote(rank(), intent.dp_size, intent.capture_tokens, (), None,
                               ("eligibility_error:" + type(exc).__name__,))
            error = exc
        votes = gather(local)
        if any(any(r.startswith("eligibility_error:") for r in v.reasons) for v in votes):
            emit({"event": "replay_admission_error", "rank": rank()})
            raise RuntimeError("Flash eligibility failed on one or more ranks") from error
        agreement = agree_replay(votes, dp_size=intent.dp_size)
        batch._memt_graph_allowed = bool(
            getattr(batch, "_memt_graph_allowed", True) and agreement.allowed
        )
        batch._memt_flash_prefill_allowed = agreement.allowed
        _count(runner, "admissions" if agreement.allowed else "fallbacks")
        if replay_observer is not None:
            replay_observer({"event": "replay_admission", "rank": local.rank,
                             "allowed": agreement.allowed, "bucket": agreement.bucket,
                             "reasons": agreement.reasons, "original_counts": local.original_counts})
        return result

    @wraps(native_can_run)
    def can_run(self, batch):
        if _is_flash(self.model_runner):
            if not getattr(batch, "_memt_flash_prefill_allowed", False):
                return False
        return native_can_run(self, batch)

    model_runner_cls._prepare_eager_forward_batch = normalized
    runner_cls.can_run_graph = can_run


def _count(runner, name):
    key = "_memt_flash_" + name
    setattr(runner, key, getattr(runner, key, 0) + 1)


def replay_counters(runner):
    """On-demand host snapshot; no logging, tensor reads or synchronization.

    execute_returns counts native CPU returns even if a later observer fails.
    Submissions and returns do not verify CUDA completion or numerical quality.
    """
    names = ("backend_submissions", "execute_returns", "execution_errors", "admissions", "fallbacks")
    return {**{name: getattr(runner, "_memt_flash_" + name, 0) for name in names},
            "cuda_completion_verified": False, "numeric_qualified": False}


def _attach_replay_counter(runner):
    """Install once after capture; no per-execute backend-method replacement."""
    native = runner.backend.replay
    if getattr(native, "_memt_flash_replay_counter", False):
        raise RuntimeError("Flash backend counter already installed")
    runner._memt_flash_expected_tokens = None

    @wraps(native)
    def replay(shape_key, static_forward_batch, **kwargs):
        expected = runner._memt_flash_expected_tokens
        if expected is None or shape_key.size != expected:
            raise RuntimeError("Flash backend replay does not match admitted execution")
        result = native(shape_key, static_forward_batch, **kwargs)
        _count(runner, "backend_submissions")
        return result

    replay._memt_flash_replay_counter = True
    runner.backend.replay = replay


def _install_execute_evidence(runner_cls, emit, replay_observer=None):
    native = runner_cls.execute

    @wraps(native)
    def execute(self, batch, *args, **kwargs):
        if not _is_flash(self.model_runner):
            return native(self, batch, *args, **kwargs)
        if not (getattr(self, "_memt_flash_capture_complete", False)
                and getattr(batch, "_memt_flash_prefill_allowed", False)):
            raise RuntimeError("Flash replay lacks capture/admission evidence")
        if self._memt_flash_expected_tokens is not None:
            raise RuntimeError("Flash replay cannot be reentered")
        tokens = len(batch.input_ids)
        before = getattr(self, "_memt_flash_backend_submissions", 0)
        self._memt_flash_expected_tokens = tokens
        try:
            if replay_observer is not None:
                replay_observer({"event": "replay_enter", "tokens": tokens})
            result = native(self, batch, *args, **kwargs)
            submitted = getattr(self, "_memt_flash_backend_submissions", 0) - before
            if submitted < 1:
                raise RuntimeError("Flash runner returned without the admitted backend replay")
            _count(self, "execute_returns")
            if replay_observer is not None:
                replay_observer({"event": "replay_return", "tokens": tokens,
                                 "backend_replay_count": submitted,
                                 "actual": {"backend": "breakable", "status": "replay_submitted"},
                                 "cuda_synchronized": False, "numeric_qualified": False})
            return result
        except Exception as error:
            _count(self, "execution_errors")
            emit({"event": "replay_error", "error_type": type(error).__name__})
            raise
        finally:
            self._memt_flash_expected_tokens = None

    runner_cls.execute = execute


def install_worker_hooks(intent, runner_cls, attention_cls, model_runner_cls, *,
                         gather, rank, emit, synchronize, memory, replay_observer=None):
    """Install before constructing any Flash PrefillCudaGraphRunner.

    gather must all-gather Python objects on the full TP=DP4 CPU group in rank
    order. emit persists per-rank evidence; it must not swallow failures.
    No worker creation, CUDA allocation or capture happens during installation.
    Capture errors abort construction; the service must not publish readiness.
    synchronize is used for capture only. Optional replay_observer receives host
    events and owns any diagnostic synchronization; exceptions propagate.
    """
    explicit_capture_config(intent)
    if replay_observer is not None and not callable(replay_observer):
        raise TypeError("replay_observer must be callable or None")
    if getattr(runner_cls, "_memt_flash_capture_gate", False):
        raise RuntimeError("Flash capture gate already installed")
    _install_metadata_hook(attention_cls)
    _install_capture_hook(intent, runner_cls, gather, emit, synchronize, memory, replay_observer)
    _install_replay_consensus(intent, runner_cls, model_runner_cls, gather, rank, emit, replay_observer)
    _install_execute_evidence(runner_cls, emit, replay_observer)
    runner_cls._memt_flash_capture_gate = True


def install_native_worker(intent, *, emit, replay_observer=None):
    """Production worker wiring; call on all four workers before model loading.

    Parent passes explicit_capture_config() in prefill JSON and calls
    validate_capture_request() after ServerArgs resolution. This function is
    intentionally not auto-enabled by importing the module.
    """
    if replay_observer is not None and not callable(replay_observer):
        raise TypeError("replay_observer must be callable or None")
    import torch
    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.runtime_context import get_parallel

    def gather(value):
        output = [None] * intent.dp_size
        torch.distributed.all_gather_object(output, value, group=get_world_group().cpu_group)
        return output

    def memory():
        free, total = torch.cuda.mem_get_info()
        return {"free_bytes": free, "total_bytes": total,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    def ranked_emit(event):
        emit({**event, "dp_rank": get_parallel().attn_dp_rank})

    def ranked_observer(event):
        replay_observer({**event, "dp_rank": get_parallel().attn_dp_rank})

    install_worker_hooks(intent, PrefillCudaGraphRunner, DeepseekV4AttnBackend, ModelRunner,
                         gather=gather, rank=lambda: get_parallel().attn_dp_rank,
                         emit=ranked_emit, synchronize=torch.cuda.synchronize, memory=memory,
                         replay_observer=ranked_observer if replay_observer is not None else None)
