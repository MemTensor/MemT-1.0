"""Evaluate the actual patched native owner expression without loading CUDA."""

import ast
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("strict", [False, True])
def test_native_capture_owners_independent_of_math_policy(strict):
    checkout = os.environ.get("MEMT_TEST_SGLANG_SOURCE")
    if not checkout:
        pytest.skip("Set MEMT_TEST_SGLANG_SOURCE to a patched source checkout")
    path = Path(checkout) / "python/sglang/srt/models/deepseek_v4.py"
    tree = ast.parse(path.read_text())
    model = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Model"
    )
    init = next(
        n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assignment = next(
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_pin_decode_graph_owners"
            for t in n.targets
        )
    )
    expression = compile(ast.Expression(assignment.value), str(path), "eval")
    for requested in (False, True):
        values = {
            "SGLANG_DSV4_PIN_DECODE_INTERMEDIATES": requested,
            "SGLANG_DSV4_DETERMINISTIC_PROFILE": strict,
        }
        assert eval(expression, {"get_bool_env_var": values.__getitem__}) == requested
