"""Check native policy expressions, not just the adapter's interpretation."""

import ast
import os
from pathlib import Path

import pytest

FLAGS = (
    "SGLANG_DSV4_DETERMINISTIC_PROFILE",
    "SGLANG_DSV4_ISOLATE_PREFILL_DECODE",
    "SGLANG_DSV4_MOE_COMBINE_FP32",
)
FILES = (
    "kernels/ops/layernorm/mhc.py",
    "srt/managers/scheduler.py",
    "srt/models/deepseek_v2.py",
    "srt/server_args.py",
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("0", False),
        ("false", False),
        (None, False),
    ],
)
def test_native_profile_booleans_match_adapter(monkeypatch, value, expected):
    from memt_sglang.runtime.features import native_flag

    checkout = os.environ.get("MEMT_TEST_SGLANG_SOURCE")
    if not checkout:
        pytest.skip("Set MEMT_TEST_SGLANG_SOURCE to a patched source checkout")
    for flag in FLAGS:
        if value is None:
            monkeypatch.delenv(flag, raising=False)
        else:
            monkeypatch.setenv(flag, value)
        assert native_flag(flag) == expected
    checked = set()
    for relative in FILES:
        path = Path(checkout) / "python/sglang" / relative
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Compare):
                continue
            source = ast.unparse(node)
            if "os.environ.get" not in source or not any(f in source for f in FLAGS):
                continue
            actual = eval(compile(ast.Expression(node), str(path), "eval"), {"os": os})
            assert actual == expected, (relative, source, value)
            checked.add(relative)
    assert checked == set(FILES)


@pytest.mark.parametrize("width,splits", [(8192, "32"), (16384, "64"), (28672, "7")])
def test_native_fixed_partition_has_no_model_shape_allowlist(
    monkeypatch, width, splits
):
    checkout = os.environ.get("MEMT_TEST_SGLANG_SOURCE")
    if not checkout:
        pytest.skip("Set MEMT_TEST_SGLANG_SOURCE to a patched source checkout")
    path = Path(checkout) / "python/sglang/kernels/ops/layernorm/mhc.py"
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_compute_num_split_for_mhc_pre"
    )
    # The configured partition must not query batch/device heuristics. CUDA
    # kernel legality and numerical behavior are validated separately on GPU.
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    monkeypatch.setenv("SGLANG_DSV4_DETERMINISTIC_PROFILE", "true")
    monkeypatch.setenv("SGLANG_DSV4_MHC_FIXED_SPLITS", splits)
    assert namespace[function.name](33, width) == int(splits)
    monkeypatch.setenv("SGLANG_DSV4_MHC_FIXED_SPLITS", "0")
    with pytest.raises(ValueError, match="positive"):
        namespace[function.name](33, width)
