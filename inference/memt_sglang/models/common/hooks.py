"""Shared memory forward lifecycle."""

import os

# Warmup shapes are a cost/coverage trade, not a property of the model: a shape
# that is never warmed compiles on its first real use, inside a decode step, in
# front of a request. The defaults below cover what the released profiles
# exercise, and the variables exist because a deployment with a different
# concurrency ceiling or context budget has a different set of "first real uses"
# and should not have to edit this file to hit them.
WARMUP_DECODE_VARIABLE = "MEMT_WARMUP_DECODE_BATCHES"
WARMUP_PREFILL_VARIABLE = "MEMT_WARMUP_PREFILL_LENGTHS"
_DEFAULT_DECODE_CEILING = 32
_DEFAULT_PREFILL_LENGTHS = (512, 2048)


def _parse_shape_list(variable, default):
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return default
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdecimal() or int(item) < 1:
            raise ValueError(
                f"{variable} takes a comma list of positive integers; got {item!r}"
            )
        values.append(int(item))
    if not values:
        raise ValueError(f"{variable} was set but lists no shape")
    return tuple(sorted(set(values)))


def _decode_ladder(maximum):
    """Powers of two up to `maximum`, plus `maximum` itself.

    Derived rather than listed so that raising the server's concurrency ceiling
    also raises what gets warmed. The previous fixed list stopped at 32 and was
    additionally clamped to 32, so a server admitting 128 concurrent requests
    compiled batches 64 and 128 on their first real use -- inside a decode step,
    with a request waiting. At the released ceiling of 32 this returns exactly
    the list it replaces, so measured configurations are unaffected.
    """
    sizes, size = [], 1
    while size < maximum:
        sizes.append(size)
        size *= 2
    sizes.append(maximum)
    return tuple(sizes)


def _warmup_decode_batches(server_args, dp_size):
    """Batch sizes to warm, derived from what the server can actually admit.

    max_running_requests is the global admission budget. DP attention divides
    it across ranks; round up to cover the largest local shard.
    """
    requested = server_args.max_running_requests or _DEFAULT_DECODE_CEILING
    maximum = max(1, (requested + dp_size - 1) // dp_size)
    configured = _parse_shape_list(WARMUP_DECODE_VARIABLE, None)
    if configured is None:
        return list(_decode_ladder(maximum))
    batches = [n for n in configured if n <= maximum]
    # A ceiling below the smallest configured batch would leave decode
    # completely cold, which is the failure this exists to prevent.
    return batches or [min(maximum, min(configured))]


def _warmup_prefill_lengths(server_args):
    """Prefill lengths to warm, capped by the context the server will serve.

    Warmup allocates `length` hidden rows at once, so a length beyond the
    configured context length would only ever exercise a shape that cannot
    occur.
    """
    configured = _parse_shape_list(WARMUP_PREFILL_VARIABLE, _DEFAULT_PREFILL_LENGTHS)
    context = getattr(server_args, "context_length", None)
    if not context:
        return configured
    kept = [n for n in configured if n <= context]
    return kept or [min(context, min(configured))]


class MemoryHooks:
    def setup_execution(self, *, dp_attention=False):
        from memt_sglang.runtime.features import enabled

        self.memory_decode_graph = None
        self.dp_attention = dp_attention
        if enabled("MEMORY_DECODE_GRAPH"):
            from memt_sglang.runtime.memory_graph import (
                MemoryDecodeGraph,
                install_memory_capture,
            )

            self.memory_decode_graph = MemoryDecodeGraph(self)
            install_memory_capture()

        from memt_sglang.runtime.features import commit_chunk_policy

        if commit_chunk_policy() != "atomic":
            # The partial scheduler owns admission; the atomic commit guard is
            # its serialized counterpart and the two must not both install.
            if not enabled("PARTIAL_NATIVE_REFERENCE"):
                from memt_sglang.runtime.partial_scheduler import (
                    install_partial_prefill,
                )

                install_partial_prefill(
                    chunk_tokens=next(iter(self.blocks.values())).chunk_tokens
                )
        elif enabled("QUERY_CHUNKS"):
            from memt_sglang.runtime.query_chunks import install_atomic_commit_guard

            install_atomic_commit_guard()
        prefill, decode = enabled("PREFILL_GRAPH"), enabled("DECODE_GRAPH")
        self.graph_enabled = prefill or decode
        if not self.graph_enabled:
            return lambda function: function
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            eager_on_graph,
        )

        from memt_sglang.runtime.decode_graph import (
            install_decode_query_guard,
            install_dp_query_consensus,
            install_logits_output_bridge,
        )
        from memt_sglang.runtime.prefill_graph import install_prefill_query_guard

        install_logits_output_bridge()
        if dp_attention:
            install_dp_query_consensus()
        if prefill:
            install_prefill_query_guard()
        if decode:
            install_decode_query_guard()

        def decorate(function):
            eager = eager_on_graph(True)(function)

            def dispatch(*args, **kwargs):
                if (
                    self.memory_decode_graph is not None
                    and self.memory_decode_graph.capture_active
                ):
                    return function(*args, **kwargs)
                return eager(*args, **kwargs)

            return dispatch

        return decorate

    def warmup_memory(self, runner):
        """Compile actual parameters before readiness; never update state."""
        from memt_sglang.runtime.features import enabled

        if not (enabled("NANO_COMPILE") or enabled("MEMORY_DECODE_GRAPH")):
            return
        import logging
        import time

        import torch

        start = time.monotonic()
        dp = runner.server_args.dp_size if self.dp_attention else 1
        server_args = runner.server_args
        decode = _warmup_decode_batches(server_args, dp)
        prefill = _warmup_prefill_lengths(server_args)
        runtime = self.model.memt_runtime
        with torch.no_grad():
            for layer, block in self.blocks.items():
                parameter = next(block.parameters())
                for tokens, lengths in [(n, (1,) * n) for n in decode] + [
                    (n, (n,)) for n in prefill
                ]:
                    hidden = torch.zeros(
                        (tokens, block.hidden_size),
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    slots = torch.zeros(
                        len(lengths), device=parameter.device, dtype=torch.int64
                    )
                    block.packed_query_slots(
                        hidden,
                        runtime.pool.storage[layer][:, 0],
                        slots,
                        lengths=lengths,
                    )
            torch.cuda.synchronize()
        logging.getLogger(__name__).info(
            "MemT tensor warmup complete: decode batches=%s, prefill lengths=%s, "
            "elapsed=%.3fs",
            decode,
            prefill,
            time.monotonic() - start,
        )

    def forward(
        self, native_forward, input_ids, positions, forward_batch, *args, **kwargs
    ):
        runtime = getattr(self.model, "memt_runtime", None)
        if runtime is None:
            raise RuntimeError("MemT pool must be allocated before model forward")
        runtime.begin(forward_batch)
        success = False
        try:
            result = native_forward(
                input_ids, positions, forward_batch, *args, **kwargs
            )
            if self.pending:
                raise RuntimeError(
                    "Memory residual was not consumed by the native projection"
                )
            success = True
            return result
        finally:
            self.pending.clear()
            runtime.finish(success=success)
