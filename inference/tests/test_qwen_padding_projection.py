"""Gear capacity must not select the live GDN gate GEMM shape."""
import sys
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from memt_sglang.models.qwen.padding import (
    install_live_gate_projection,
    live_gate_projection,
    wrap_input_projection,
)


@pytest.mark.parametrize('valid,gear', [(511, 512), (512, 512), (513, 1024), (1023, 1024), (1536, 2048)])
def test_gate_uses_live_shape_and_zeroes_unused_rows(valid, gear):
    calls = []
    def native(x):
        calls.append(len(x))
        return x * len(x), None
    hidden = torch.ones(gear, 3)
    output = live_gate_projection(native, hidden, valid)
    assert calls == [valid]
    assert output.shape == hidden.shape
    torch.testing.assert_close(output[:valid], hidden[:valid] * valid)
    assert not torch.count_nonzero(output[valid:])


@pytest.mark.parametrize('valid', [None, True, 0, -1, 5, 1.5])
def test_invalid_live_count_fails_before_gemm(valid):
    with pytest.raises(ValueError):
        live_gate_projection(lambda x: pytest.fail('must reject'), torch.ones(4, 3), valid)


def test_eager_and_decode_keep_native_projection_delegate():
    calls = []
    layer = NS(in_proj_qkvz=lambda x: (x+1, None), in_proj_ba=lambda x: (x*len(x), None))
    def native(x):
        calls.append(x)
        return ('native-qkv', 'native-ba')
    batch = NS(forward_mode=NS(is_extend=lambda: False))
    context = [None]
    wrapped = wrap_input_projection(layer, native, lambda: context[0], lambda f: f)
    hidden = torch.ones(4, 3)
    assert wrapped(hidden) == ('native-qkv', 'native-ba')
    context[0] = NS(forward_batch=batch)
    assert wrapped(hidden) == ('native-qkv', 'native-ba')
    assert len(calls) == 2


def test_callback_reads_live_context_each_replay():
    context = [NS(forward_batch=NS(forward_mode=NS(is_extend=lambda: True), num_token_non_padded_cpu=4))]
    callbacks = []
    def eager(function):
        callbacks.append(function)
        return function
    layer = NS(in_proj_qkvz=lambda x: (x+1, None), in_proj_ba=lambda x: (x*len(x), None))
    wrapped = wrap_input_projection(layer, lambda x: pytest.fail('native fallback'), lambda: context[0], eager)
    hidden = torch.ones(4, 3)
    qkv, ba = wrapped(hidden)
    torch.testing.assert_close(qkv, hidden+1)
    torch.testing.assert_close(ba, hidden*4)
    context[0] = NS(forward_batch=NS(num_token_non_padded_cpu=3))
    replayed = callbacks[0](hidden)
    torch.testing.assert_close(replayed[:3], hidden[:3]*3)
    assert not torch.count_nonzero(replayed[3:])


@pytest.fixture
def pinned_graph_api(monkeypatch):
    """Replace only the pinned SGLang imports used by the real installer."""
    context = NS(current=None)
    callbacks = []

    def decorate(function):
        callbacks.append(function)
        return function

    eager_on_graph = Mock(return_value=decorate)
    prefix = 'sglang.srt.model_executor.runner_backend_utils.'
    graph = ModuleType(prefix + 'breakable_cuda_graph')
    graph.eager_on_graph = eager_on_graph
    piecewise = ModuleType(prefix + 'tc_piecewise_cuda_graph')
    piecewise.get_tc_piecewise_forward_context = lambda: context.current
    monkeypatch.setitem(sys.modules, graph.__name__, graph)
    monkeypatch.setitem(sys.modules, piecewise.__name__, piecewise)
    return NS(context=context, eager=eager_on_graph, callbacks=callbacks)


def projection_layer():
    return NS(
        _forward_input_proj=Mock(return_value=(object(), object())),
        in_proj_qkvz=Mock(),
        in_proj_ba=Mock(),
    )


def test_installer_wraps_only_linear_attention_in_heterogeneous_model(pinned_graph_api):
    first, second, full_attention, unrelated = [projection_layer() for _ in range(4)]
    originals = [layer._forward_input_proj for layer in (first, second, full_attention, unrelated)]
    model = NS(layers=[
        NS(linear_attn=first),
        NS(self_attn=full_attention),
        NS(linear_attn=None, self_attn=full_attention),
        unrelated,
        NS(linear_attn=second),
    ])

    install_live_gate_projection(model)

    assert first._forward_input_proj is not originals[0]
    assert second._forward_input_proj is not originals[1]
    assert full_attention._forward_input_proj is originals[2]
    assert unrelated._forward_input_proj is originals[3]
    assert not hasattr(full_attention, '_memt_live_gate_projection')
    assert not hasattr(unrelated, '_memt_live_gate_projection')
    assert len(pinned_graph_api.callbacks) == 2
    assert pinned_graph_api.eager.call_count == 2
    for call in pinned_graph_api.eager.call_args_list:
        assert call.args == (True,) and call.kwargs == {}
    for original in originals:
        original.assert_not_called()


def test_installer_is_idempotent_even_for_shared_layer(pinned_graph_api):
    layer = projection_layer()
    model = NS(layers=[NS(linear_attn=layer), NS(linear_attn=layer)])
    install_live_gate_projection(model)
    installed = layer._forward_input_proj

    install_live_gate_projection(model)

    assert layer._forward_input_proj is installed
    pinned_graph_api.eager.assert_called_once_with(True)
    assert len(pinned_graph_api.callbacks) == 1


@pytest.mark.parametrize('mode', ['eager', 'decode'])
def test_installed_projection_preserves_native_delegate_contract(pinned_graph_api, mode):
    layer = projection_layer()
    native = layer._forward_input_proj
    install_live_gate_projection(NS(layers=[NS(linear_attn=layer)]))
    if mode == 'decode':
        pinned_graph_api.context.current = NS(
            forward_batch=NS(forward_mode=NS(is_extend=lambda: False))
        )
    hidden = torch.ones(2, 3)

    assert layer._forward_input_proj(hidden) is native.return_value
    assert native.call_count == 1
    assert native.call_args.args[0] is hidden
    assert native.call_args.kwargs == {}
    layer.in_proj_qkvz.assert_not_called()
    layer.in_proj_ba.assert_not_called()

    error = RuntimeError('native projection failure')
    native.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        layer._forward_input_proj(hidden)
    assert raised.value is error


def test_installed_prefill_uses_pinned_context_and_projection_outputs(pinned_graph_api):
    layer = projection_layer()
    native = layer._forward_input_proj
    hidden = torch.ones(2, 3)
    qkv, gates = torch.empty(2, 7), torch.empty(2, 2)
    layer.in_proj_qkvz.return_value = (qkv, object())
    layer.in_proj_ba.return_value = (gates, object())
    install_live_gate_projection(NS(layers=[NS(linear_attn=layer)]))
    # Resolve the forward context at call time, after installation.
    pinned_graph_api.context.current = NS(forward_batch=NS(
        forward_mode=NS(is_extend=lambda: True), num_token_non_padded_cpu=2
    ))

    actual_qkv, actual_gates = layer._forward_input_proj(hidden)

    assert actual_qkv is qkv and actual_gates is gates
    native.assert_not_called()
    assert layer.in_proj_qkvz.call_count == 1
    assert layer.in_proj_qkvz.call_args.args[0] is hidden
    assert layer.in_proj_ba.call_count == 1
    gate_input = layer.in_proj_ba.call_args.args[0]
    assert gate_input.shape == hidden.shape
    assert gate_input.data_ptr() == hidden.data_ptr()
