"""Native attention/WO_B hooks; session ownership remains in shared runtime."""

import torch

from memt_sglang.models.common.hooks import MemoryHooks


class DeepSeekMemoryHooks(MemoryHooks):
    """Preserve adapter BF16 WO_A projection, with full-head DP-local state."""

    def __init__(self, model, blocks):
        self.model = model
        self.blocks = blocks
        self.pending = {}
        self.handles = []
        eager_memory = self.setup_execution(dp_attention=True)
        attentions = {}
        # Validate everything before registering the first block/hook.
        for layer, block in blocks.items():
            if not model.model.start_layer <= layer < model.model.end_layer:
                raise ValueError("Memory layer is outside the native model")
            attn = model.model.layers[layer].self_attn
            parameter = next(block.parameters())
            if hasattr(attn, "memt"):
                raise ValueError("Memory is already attached")
            if (
                block.hidden_size != attn.wo_b.output_size
                or block.num_heads != attn.n_heads
                or block.head_dim != attn.head_dim
                or parameter.device != attn.wo_a.weight.device
            ):
                raise ValueError(
                    "DeepSeek memory and native attention must have matching heads, dimensions and device"
                )
            attentions[layer] = attn
        for layer, attn in attentions.items():
            attn.add_module("memt", blocks[layer])
            self.handles.append(
                attn.register_forward_pre_hook(
                    eager_memory(self._prepare(layer, attn)), with_kwargs=True
                )
            )
            self.handles.append(
                attn.wo_b.register_forward_pre_hook(eager_memory(self._inject(layer)))
            )

    def _prepare(self, layer, attn):
        def hook(module, positional, kwargs):
            runtime = getattr(self.model, "memt_runtime", None)
            graph = self.memory_decode_graph
            capturing = graph is not None and graph.capture_active
            active = (
                runtime is not None
                and runtime.plan is not None
                and bool(runtime.plan.samples)
            )
            if not active and not self.graph_enabled and not capturing:
                return
            hidden = kwargs["x"]
            if len(hidden) == 0:
                return
            output = hidden.new_empty((len(hidden), attn.memt.value_dim))

            memory = (
                graph.read(layer, hidden, attn.memt)
                if capturing
                else (
                    runtime.run_layer(layer, hidden, output, attn.memt)
                    if active
                    else output.zero_()
                )
            )

            # Attention TP is 1: no head slicing and no collective on memory.
            memory = memory.view(len(hidden), attn.n_heads, attn.head_dim)
            memory = memory.reshape(len(hidden), attn.n_local_groups, -1)
            weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
            self.pending[layer] = torch.einsum("tgd,grd->tgr", memory, weight).flatten(
                1
            )

        return hook

    def _inject(self, layer):
        def hook(module, positional):
            memory = self.pending.pop(layer, None)
            if (
                self.memory_decode_graph is not None
                and self.memory_decode_graph.capture_active
                and memory is not None
            ):
                self.memory_decode_graph.injected.add((layer, len(memory)))
            return (
                positional
                if memory is None
                else (positional[0] + memory, *positional[1:])
            )

        return hook
