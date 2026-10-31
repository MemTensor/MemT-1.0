"""Shared memory forward lifecycle."""


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
            if not enabled("PARTIAL_NATIVE_REFERENCE"):
                from memt_sglang.runtime.partial_scheduler import (
                    install_partial_prefill,
                )

                install_partial_prefill()
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
        requested = runner.server_args.max_running_requests or 32
        maximum = min(32, (requested + dp - 1) // dp)
        batches = [n for n in (1, 2, 4, 8, 16, 32) if n <= maximum]
        runtime = self.model.memt_runtime
        with torch.no_grad():
            for layer, block in self.blocks.items():
                parameter = next(block.parameters())
                for tokens, lengths in [(n, (1,) * n) for n in batches] + [
                    (512, (512,)),
                    (2048, (2048,)),
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
            "MemT tensor warmup complete: decode batches=%s, prefill=[512,2048], elapsed=%.3fs",
            batches,
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
