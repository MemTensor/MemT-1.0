"""Keep the GDN gate projection's live GEMM shape under prefill gear padding."""

from functools import wraps


def live_gate_projection(project, hidden, valid):
    if type(valid) is not int or not 0 < valid <= len(hidden):
        raise ValueError("Invalid live token count for GDN gate projection")
    values, _ = project(hidden[:valid])
    if valid == len(hidden):
        return values
    padded = values.new_zeros((len(hidden), *values.shape[1:]))
    padded[:valid].copy_(values)
    return padded


def wrap_input_projection(layer, native, context, eager):
    @eager
    def gates(hidden):
        # Resolve at replay time. Capturing the original ForwardBatch here
        # would retain the capture gear's token count for every later request.
        batch = context().forward_batch
        return live_gate_projection(layer.in_proj_ba, hidden, batch.num_token_non_padded_cpu)

    @wraps(native)
    def project(hidden):
        current = context()
        if current is None or not current.forward_batch.forward_mode.is_extend():
            return native(hidden)
        # Keep the wide QKV projection captured. The narrow BA projection is
        # sensitive to M=511 versus512 on the pinned BF16 CUDA path. Execute it
        # on live rows at the main-stream boundary, then restore gear capacity.
        # Entering an eager graph break on the native BA side stream is invalid.
        qkv, _ = layer.in_proj_qkvz(hidden)
        return qkv, gates(hidden)

    return project


def install_live_gate_projection(model):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    for decoder in model.layers:
        layer = getattr(decoder, "linear_attn", None)
        if layer is None or getattr(layer, "_memt_live_gate_projection", False):
            continue
        layer._forward_input_proj = wrap_input_projection(
            layer, layer._forward_input_proj, get_tc_piecewise_forward_context,
            eager_on_graph(True),
        )
        layer._memt_live_gate_projection = True
