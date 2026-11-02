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
