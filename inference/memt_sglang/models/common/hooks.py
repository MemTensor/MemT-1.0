"""Shared memory forward lifecycle."""


class MemoryHooks:
    def setup_execution(self, *, dp_attention=False):
        from memt_sglang.runtime.features import enabled

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
            install_prefill_query_guard(exact_tokens=not dp_attention)
        if decode:
            install_decode_query_guard()
        return eager_on_graph(True)

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
