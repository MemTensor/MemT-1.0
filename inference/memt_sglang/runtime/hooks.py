"""SGLang allocation, scheduler and tokenizer hooks for eager serving.

All source line references below refer to SGLang v0.5.16,
fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1. Importing this module is CPU-safe;
only install_sglang_hooks() and the worker measurements import SGLang.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps

import torch

from memt_sglang import require_sglang
from memt_sglang.runtime.config import RuntimeConfig
from memt_sglang.runtime.executor import MemoryRuntime
from memt_sglang.runtime.state import StatePool, state_bytes_per_session

logger = logging.getLogger(__name__)
_GIB = 1 << 30


@dataclass(frozen=True)
class PoolBudget:
    pool_bytes: int
    remaining_bytes: int
    max_sessions: int


def calculate_budget(
    *,
    free_gib: float,
    pre_model_load_gib: float,
    mem_fraction_static: float,
    session_bytes: int,
    config: RuntimeConfig,
) -> PoolBudget:
    """Apply SGLang's eager budget; the session bound does not guarantee min KV."""
    # mem_cache/kv_cache_configurator.py:1527-1548,1570: inputs are GiB.
    # The eager path has no post-capture reserve. Qwen's native Mamba budget is
    # subsequently handled by SGLang (1549-1550), hence the post-KV token check.
    available_bytes = int(
        (free_gib - pre_model_load_gib * (1 - mem_fraction_static)) * _GIB
    )
    from .features import candidate_slots_per_transaction

    private_slots = (
        config.max_inflight_commits_per_rank * candidate_slots_per_transaction()
    )
    pool_bytes = (config.max_sessions_per_rank + private_slots) * session_bytes
    # Leave at least one byte for KV; equality is insufficient. This is an upper
    # bound before native GDN reservation, allocator rounding and min_kv_tokens.
    max_sessions = max(
        0,
        (available_bytes - 1) // session_bytes - private_slots,
    )
    return PoolBudget(pool_bytes, available_bytes - pool_bytes, max_sessions)


def _measure_free_gib(runner) -> float:
    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.utils.common import get_available_gpu_memory

    world = get_world_group()
    # kv_cache_configurator.py:1531-1536 uses WORLD, not attention TP. This
    # includes every DeepSeek DP-attention rank even though attention TP is 1.
    # utils/common.py:392-412,531-538 empties cache, reduces MIN, returns GiB.
    return get_available_gpu_memory(
        runner.device,
        runner.gpu_id,
        distributed=world.world_size > 1,
        empty_cache=True,
        cpu_group=world.cpu_group,
    )


def _check_rank_layout(layout: tuple) -> None:
    from sglang.srt.distributed.parallel_state import get_world_group

    world = get_world_group()
    if world.world_size > 1:
        layouts = [None] * world.world_size
        torch.distributed.all_gather_object(layouts, layout, group=world.cpu_group)
        if any(other != layout for other in layouts):
            raise ValueError(
                "MemT state geometry and runtime capacity must match on all ranks"
            )


class AllocationStatus(StrEnum):
    """One attempt per runner: reuse READY; reject reentry or partial failure.

    Native KV allocation may leave resources behind on failure. Retrying this
    hook would allocate MemT state twice, so FAILED requires a worker restart.
    """

    ALLOCATING = "allocating"
    READY = "ready"
    FAILED = "failed"


@dataclass
class _Allocation:
    signature: tuple
    memory_pool_config: object
    status: str = AllocationStatus.ALLOCATING
    result: object = None


def install_alloc_memory_pool_hook(runner_class: type) -> None:
    """Install once on a ModelRunner class; fake classes support CPU tests.

    MemT model adapters provide memt_state_shapes: {layer_id: state_shape}.
    This is an internal model contract, not an assumed HF field. Non-MemT
    models and draft workers retain the native method's behavior.
    """
    require_sglang()
    original = runner_class.alloc_memory_pool
    if getattr(original, "_memt_pool_hook", False):
        return

    @wraps(original)
    def alloc_memory_pool(self, memory_pool_config=None):
        # model_executor/model_runner.py:728-749: preserve the optional argument,
        # allocate before configure(), inspect max_total_num_tokens afterwards.
        # managers/tp_worker.py:374-378 calls this on target and draft runners.
        if self.is_draft_worker:
            return original(self, memory_pool_config)
        shapes = getattr(self.model, "memt_state_shapes", None)
        hf_config = self.model_config.hf_config
        if shapes is None:
            memt = (
                hf_config.get("memt")
                if isinstance(hf_config, Mapping)
                else getattr(hf_config, "memt", None)
            )
            if memt is not None:
                raise ValueError(
                    "MemT model did not provide memt_state_shapes before pool allocation"
                )
            return original(self, memory_pool_config)
        if not isinstance(shapes, Mapping):
            raise ValueError("memt_state_shapes must map layer IDs to state shapes")
        session_bytes = state_bytes_per_session(shapes)
        shapes = {layer: tuple(shape) for layer, shape in shapes.items()}
        config = RuntimeConfig.from_hf_config(hf_config)
        # The configurator receives server_args (model_runner.py:513-529).
        # Use its resolved fraction, not an independently computed default.
        fraction = self.server_args.mem_fraction_static
        from .features import commit_chunk_policy

        layout = (
            tuple(sorted(shapes.items())),
            config,
            fraction,
            commit_chunk_policy(),
        )
        signature = (layout, self.device, self.gpu_id)
        previous = getattr(self, "_memt_pool_allocation", None)
        if previous is not None:
            if previous.status != AllocationStatus.READY:
                raise RuntimeError(
                    "MemT pool initialization is incomplete or failed; restart the worker"
                )
            if (
                previous.signature != signature
                or previous.memory_pool_config is not memory_pool_config
            ):
                raise RuntimeError(
                    "Cannot reconfigure initialized MemT/KV pools; restart the worker"
                )
            return previous.result
        _check_rank_layout(layout)
        allocation = _Allocation(signature, memory_pool_config)
        self._memt_pool_allocation = allocation
        try:
            free_before = _measure_free_gib(self)
            budget = calculate_budget(
                free_gib=free_before,
                pre_model_load_gib=self.pre_model_load_memory,
                mem_fraction_static=fraction,
                session_bytes=session_bytes,
                config=config,
            )
            if budget.remaining_bytes <= 0:
                raise MemoryError(
                    f"MemT pool needs {budget.pool_bytes} bytes and leaves no KV budget; "
                    f"max_sessions_per_rank must be <= {budget.max_sessions} with "
                    f"max_inflight_commits_per_rank={config.max_inflight_commits_per_rank}. "
                    "This bound still excludes native GDN reservation and min_kv_tokens."
                )
            if getattr(self.model, "memt_runtime", None) is not None:
                raise RuntimeError(
                    "MemT model already owns a runtime before pool allocation"
                )
            # model_runner.py:258-259 separates device type and GPU ordinal.
            device = (
                torch.device(self.device, self.gpu_id)
                if self.device == "cuda"
                else torch.device(self.device)
            )
            pool = StatePool(
                shapes,
                max_sessions_per_rank=config.max_sessions_per_rank,
                max_inflight_commits_per_rank=config.max_inflight_commits_per_rank,
                device=device,
            )
            self.model.memt_runtime = MemoryRuntime(pool)
            free_after = _measure_free_gib(self)
            logger.info(
                "MemT state pool: %d bytes, free before=%.6f GiB, free after=%.6f GiB "
                "(world minimum); eager KV estimate=%d bytes",
                pool.nbytes,
                free_before,
                free_after,
                budget.remaining_bytes,
            )
            result = original(self, memory_pool_config)
            # model_runner.py:737 obtains the final native capacity. This catches
            # GDN reservations, alignment and explicit max-total-tokens limits.
            tokens = self.max_total_num_tokens
            from .features import validate_native_kv_capacity

            validate_native_kv_capacity(
                tokens, getattr(self.server_args, "max_total_tokens", None)
            )
            if tokens < config.min_kv_tokens:
                raise MemoryError(
                    f"SGLang max_total_num_tokens={tokens} is below "
                    f"memt_runtime.min_kv_tokens={config.min_kv_tokens}; "
                    "reduce max_sessions_per_rank and check --max-total-tokens."
                )
            hooks = getattr(self.model, "_memory_hooks", None)
            if hooks is not None:
                hooks.warmup_memory(self)
            allocation.result = result
            allocation.status = AllocationStatus.READY
            return result
        except BaseException:
            # Native KV initialization can have partially allocated its own pools.
            # Never retry/reallocate in this worker or silently shrink capacity.
            allocation.status = AllocationStatus.FAILED
            raise

    alloc_memory_pool._memt_pool_hook = True
    runner_class.alloc_memory_pool = alloc_memory_pool


def install_sglang_hooks() -> None:
    """Called by the external model package, after the strict version check."""
    require_sglang()
    from sglang.srt.model_executor.model_runner import ModelRunner

    install_alloc_memory_pool_hook(ModelRunner)


def install_tokenizer_control(manager):
    """0.5.16 instance dispatcher interception; correlate only our tagged replies."""
    from sglang.srt.managers.io_struct import GetInternalStateReqOutput

    from memt_sglang.runtime.control import PREFIX, ControlChannel

    if hasattr(manager, "memt_control"):
        return
    channel = manager.memt_control = ControlChannel(manager)
    original = manager._result_dispatcher

    def dispatch(message):
        if isinstance(message, GetInternalStateReqOutput) and (
            message.rid or ""
        ).startswith(PREFIX):
            return channel.receive(message)
        return original(message)

    manager._result_dispatcher = dispatch


def install_scheduler_hooks():
    """Pinned control, admission and opt-in query scheduling hooks."""
    require_sglang()
    import json

    from sglang.srt.managers.io_struct import AbortReq, GetInternalStateReqOutput
    from sglang.srt.managers.scheduler import Scheduler

    from memt_sglang.protocol import SessionError, is_memt_rid
    from memt_sglang.runtime.control import PREFIX, worker_control

    if getattr(Scheduler, "_memt_control_installed", False):
        return
    from .features import overlap_phases
    from .query_coalescing import config_from_env, install_query_coalescing

    coalescing = config_from_env()
    if coalescing is not None:
        install_query_coalescing(Scheduler, coalescing)
    prefill, decode = overlap_phases()
    if prefill or decode:
        from .decode_overlap import install_phase_overlap

        install_phase_overlap(Scheduler, prefill=prefill, decode=decode)
    native_control = Scheduler.get_internal_state
    native_generate = Scheduler.handle_generate_request

    def control(self, request):
        if not (request.rid or "").startswith(PREFIX):
            return native_control(self, request)
        envelope = json.loads(request.rid[len(PREFIX) :])
        rank = self.ps.dp_rank or 0
        result = {"rank": rank, "ok": True, "result": None}
        if rank == envelope["rank"]:
            # Eager scheduling serializes CPU control and forward; synchronize
            # just this rank's stream before freeing/copying memory slots.
            self.forward_stream.synchronize()
            try:
                with torch.cuda.stream(self.forward_stream):
                    result["result"] = worker_control(self, envelope)
                self.forward_stream.synchronize()
            except (ValueError, SessionError, OSError) as error:
                result.update(
                    ok=False, error=str(error), error_type=type(error).__name__
                )
            if self.ps.attn_tp_size > 1:
                values = [None] * self.ps.attn_tp_size
                torch.distributed.all_gather_object(
                    values, result, group=self.attn_tp_cpu_group
                )
                errors = [value for value in values if not value["ok"]]
                if errors:
                    result = errors[0]
        return GetInternalStateReqOutput(rid=request.rid, internal_state=result)

    def generate(self, request, *args, **kwargs):
        if is_memt_rid(request.rid):
            prepared = (
                self.tp_worker.model_runner.model.memt_runtime.worker_sessions.requests
            )
            if request.rid not in prepared:
                return AbortReq(rid=request.rid)
        return native_generate(self, request, *args, **kwargs)

    Scheduler.get_internal_state = control
    Scheduler.handle_generate_request = generate
    Scheduler._memt_control_installed = True
