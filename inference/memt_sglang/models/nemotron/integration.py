"""Inject memory before native attention o_proj; retain Mamba/MoE and TP."""

from contextlib import contextmanager

from sglang.srt.runtime_context import get_parallel

from memt_sglang.models.common.hooks import MemoryHooks


class NemotronMemoryHooks(MemoryHooks):
    def __init__(self, model, blocks):
        self.model, self.blocks, self.pending, self.handles = model, blocks, {}, []
        eager_memory = self.setup_execution(dp_attention=False)
        parallel = get_parallel()
        self.tp_rank, self.tp_size = parallel.attn_tp_rank, parallel.attn_tp_size
        for index, block in blocks.items():
            mixer = model.model.layers[index].mixer
            weight, parameter = mixer.o_proj.weight, next(block.parameters())
            if (
                not 0 <= self.tp_rank < self.tp_size
                or mixer.num_heads * self.tp_size != block.num_heads
                or mixer.head_dim != block.head_dim
                or hasattr(mixer, "memt")
                or parameter.dtype != weight.dtype
                or parameter.device != weight.device
            ):
                raise ValueError(
                    "Nemotron attention/memory partition, dtype or device mismatch"
                )
        for index, block in blocks.items():
            mixer = model.model.layers[index].mixer
            mixer.add_module("memt", block)
            self.handles.append(
                mixer.qkv_proj.register_forward_pre_hook(
                    eager_memory(self._prepare(index, mixer)), with_kwargs=True
                )
            )
            self.handles.append(
                mixer.o_proj.register_forward_pre_hook(
                    eager_memory(self._inject(index))
                )
            )

    @contextmanager
    def decode_context(self, batch):
        """Supply the native Mamba split-op context during decode graph execution.

        SGLang 0.5.16 sets this context for prefill but not breakable decode.
        The Mamba operator itself and its cache updates remain native.
        """
        from sglang.srt.model_executor.model_runner_components.layer_setup import (
            compute_attention_and_moe_layers,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
            get_tc_piecewise_forward_context,
            set_tc_piecewise_forward_context,
        )

        if get_tc_piecewise_forward_context() is not None:
            yield
            return
        if not hasattr(self, "_decode_layers"):
            self._decode_layers = compute_attention_and_moe_layers(self.model.model)
        attention, moe, fusions, indexers, companions = self._decode_layers
        with set_tc_piecewise_forward_context(
            batch,
            attention,
            None,
            moe,
            fusions,
            dsa_indexers=indexers,
            mha_companion_layers=companions,
            num_tokens=len(batch.input_ids),
            raw_num_tokens=len(batch.input_ids),
        ):
            yield

    def _prepare(self, index, mixer):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["input_"]
            residual = self.captured_memory(index, hidden, mixer.memt)
            if residual is not None:
                heads = residual.view(len(hidden), mixer.memt.num_heads, mixer.head_dim)
                start = self.tp_rank * mixer.num_heads
                self.pending[index] = heads[:, start : start + mixer.num_heads].reshape(
                    len(hidden), -1
                )
                return
            runtime = getattr(self.model, "memt_runtime", None)
            active = (
                runtime is not None
                and runtime.plan is not None
                and bool(runtime.plan.samples)
            )
            if not active and not self.graph_enabled:
                return
            if len(hidden) == 0:
                return
            output = hidden.new_empty((len(hidden), mixer.memt.value_dim))
            residual = (
                runtime.run_layer(index, hidden, output, mixer.memt)
                if active
                else output.zero_()
            )
            heads = residual.view(len(hidden), mixer.memt.num_heads, mixer.head_dim)
            start = self.tp_rank * mixer.num_heads
            self.pending[index] = heads[:, start : start + mixer.num_heads].reshape(
                len(hidden), -1
            )

        return hook

    def _inject(self, index):
        def hook(module, args):
            residual = self.pending.pop(index, None)
            self.note_captured_injection(index, residual)
            return args if residual is None else (args[0] + residual, *args[1:])

        return hook
